"""Classic SeqSeg batch pipeline (JSON seeds or centerline-based init)."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import SimpleITK as sitk

from seqseg.modules import initialization as init
from seqseg.modules import sitk_functions as sf
from seqseg.modules import vtk_functions as vf
from seqseg.modules.assembly import calc_centerline_global
from seqseg.modules.capping import cap_surface
from seqseg.modules.simvascular import write_simvascular_proj
from seqseg.modules.tracing import trace_centerline_from_context, TracingContext
from seqseg.pipeline.directories import create_case_directories


def run_classic_batch(
    *,
    dir_output0: str,
    data_dir: str,
    dataset: str,
    dir_model_weights: str,
    fold: str,
    img_format: str,
    scale: float,
    test_name: str,
    pt_centerline: int,
    num_seeds: int,
    testing_samples: Sequence,
    directory_data: str,
    start: int,
    stop: int,
    global_config: Mapping[str, Any],
    unit: str,
    max_step_size: int,
    max_n_branches: int,
    max_n_steps_per_branch: int,
    write_samples: bool,
    take_time: bool,
    calc_global_centerline: bool,
    cap_surface_cent: bool,
    assembly_threshold: float,
    resample_spacing: Optional[Sequence[float]],
    start_time_global: float,
) -> None:
    """Run classic tracing for ``testing_samples[start:stop]``."""
    assembly_spacing_factor = global_config.get("ASSEMBLY_SPACING_FACTOR", 1.0)
    if assembly_spacing_factor <= 0:
        raise ValueError("assembly_spacing_factor must be > 0")

    for i, test_case in enumerate(testing_samples[start:stop]):
        print(
            f"\nProcessing test case {i + start + 1} of {len(testing_samples)}: "
            f"{test_case}"
        )

        dir_output, dir_image, dir_seg, dir_cent, case, idx, json_file_present = (
            init.process_init(
                test_case, directory_data, dir_output0, img_format, test_name
            )
        )

        create_case_directories(dir_output, write_samples)
        if resample_spacing is not None:
            dir_image, dir_seg = sf.maybe_resample_volume_paths(
                dir_image,
                dir_seg,
                resample_spacing,
                dir_output,
                case,
                img_format,
            )
        vf.write_image_as_vti(
            dir_image,
            os.path.join(dir_output, "simvascular", "Images", f"{case}.vti"),
        )
        write_simvascular_proj(os.path.join(dir_output, "simvascular"))

        potential_branches, initial_seeds = init.initialization(
            json_file_present,
            test_case,
            dir_output,
            dir_cent,
            directory_data,
            unit,
            pt_centerline,
            num_seeds,
            write_samples,
        )

        ref_image = sitk.ReadImage(dir_image)
        sf.validate_potential_branches_image_bounds(
            ref_image, potential_branches, case=case, image_path=dir_image
        )

        if not global_config["DEBUG"]:
            sys.stdout = open(dir_output + "/out.txt", "w")
        else:
            print("\nStart tracking with debug mode on")

        if json_file_present:
            print("\nWe got seed point from json file")
        else:
            print("\nWe did not get seed point from json file")
        print(test_case)
        print(f"Number of initial points: {len(potential_branches)}")
        print(f"Time is: {time.time() - start_time_global:.2f} sec")

        ctx = TracingContext(
            output_folder=dir_output,
            image_file=dir_image,
            case=case,
            model_folder=dir_model_weights,
            fold=fold,
            potential_branches=potential_branches,
            max_step_size=max_step_size,
            max_n_branches=max_n_branches,
            max_n_steps_per_branch=max_n_steps_per_branch,
            global_config=global_config,
            unit=unit,
            scale=scale,
            seg_file=dir_seg,
            write_samples=write_samples,
        )
        tr = trace_centerline_from_context(ctx)
        centerlines = tr.centerlines
        surfaces = tr.surfaces
        points = tr.points
        inside_pts = tr.inside_pts
        assembly_obj = tr.assembly
        vessel_tree = tr.vessel_tree
        n_steps_taken = tr.n_steps_taken

        print(
            "\nTotal calculation time is:"
            + f"{((time.time() - start_time_global) / 60):.2f} min\n"
        )

        if take_time:
            vessel_tree.time_analysis()

        if global_config["TREE_ANALYSIS"]:
            vessel_tree.create_tree_polydata_v1(dir_output)
            vessel_tree.create_tree_polydata_v2(dir_output)
            vessel_tree.plot_radius_distribution(dir_output)

        assembly = assembly_obj.assembly
        n_udpates = None
        if write_samples:
            n_udpates = assembly_obj.get_n_updates_image()

        if assembly_spacing_factor != 1.0:
            target_spacing = [
                spacing * assembly_spacing_factor for spacing in assembly.GetSpacing()
            ]
            print(f"Resampling final assembly to spacing: {target_spacing}")
            assembly = sf.resample_to_spacing(assembly, target_spacing, is_label=False)
            if write_samples:
                n_udpates = sf.resample_to_spacing(
                    n_udpates, target_spacing, is_label=True
                )

        assembly_binary = sitk.BinaryThreshold(
            assembly,
            lowerThreshold=assembly_threshold,
            upperThreshold=1,
        )

        sitk.WriteImage(
            assembly_binary,
            dir_output + "/" + case + "_binary_seg_" + test_name + "_" + str(idx) + ".mha",
        )
        if write_samples:
            sitk.WriteImage(
                assembly,
                dir_output + "/" + case + "_prob_seg_" + test_name + "_" + str(idx) + ".mha",
            )
            sitk.WriteImage(
                n_udpates,
                dir_output + "/" + case + "_n_updates_" + test_name + "_" + str(idx) + ".mha",
            )

        assembly_obj.calc_ratio_updates()

        assembly_binary = sf.keep_component_seeds(assembly_binary, initial_seeds)

        sitk.WriteImage(
            assembly_binary,
            dir_output0
            + "/"
            + case
            + "_segmentation_"
            + test_name
            + "_"
            + str(n_steps_taken)
            + "_steps"
            + ".mha",
        )

        assembly_surface = vf.evaluate_surface(assembly_binary, 1)

        vf.write_vtk_polydata(
            assembly_surface,
            dir_output + "/" + case + "_surface_mesh_nonsmooth_" + test_name + "_"
            + str(n_steps_taken) + "_steps" + ".vtp",
        )

        surface_smooth = vf.taubin_smooth_polydata(
            assembly_surface, it=75, mu1=0.5, mu2=0.51
        )
        surface_smooth = vf.compute_polydata_normals(surface_smooth)
        vf.write_vtk_polydata(
            surface_smooth,
            dir_output0 + "/" + case + "_surface_mesh_" + test_name + "_"
            + str(n_steps_taken) + "_steps" + ".vtp",
        )

        if len(centerlines) > 0:
            final_surface = vf.appendPolyData(surfaces)
            final_centerline = vf.appendPolyData(centerlines)
            final_points = vf.appendPolyData(points)

            vf.write_vtk_polydata(
                final_surface,
                dir_output + "/all_" + case + "_" + test_name + "_" + str(idx) + "_"
                + str(n_steps_taken) + "_surfaces.vtp",
            )
            vf.write_vtk_polydata(
                final_centerline,
                dir_output + "/all_" + case + "_" + test_name + "_" + str(idx) + "_"
                + str(n_steps_taken) + "_centerlines.vtp",
            )
            vf.write_vtk_polydata(
                final_points,
                dir_output + "/all_" + case + "_" + test_name + "_" + str(idx) + "_"
                + str(n_steps_taken) + "_points.vtp",
            )

        if calc_global_centerline:
            global_centerline, targets, success = calc_centerline_global(
                assembly_binary,
                initial_seeds,
                nr_seeds=len(initial_seeds),
                merge_method=global_config.get("CENTERLINE_MERGE_METHOD", "clean"),
            )

            if success or len(targets) > 0:
                vf.write_vtk_polydata(
                    global_centerline,
                    dir_output0 + "/" + case + "_centerline_" + test_name + "_"
                    + str(n_steps_taken) + "_steps" + ".vtp",
                )

                targets_pd = vf.points2polydata([target.tolist() for target in targets])
                vf.write_vtk_polydata(
                    targets_pd,
                    dir_output + "/" + case + "_" + test_name + "_" + str(idx) + "_"
                    + str(n_steps_taken) + "_targets.vtp",
                )

                if cap_surface_cent:
                    capped_surface, capped_seg = cap_surface(
                        pred_surface=assembly_surface,
                        centerline=global_centerline,
                        pred_seg=assembly_binary,
                        file_name=case,
                        outdir=dir_output,
                        targets=targets,
                    )

                    vf.write_vtk_polydata(
                        capped_surface,
                        dir_output + "/" + case + "_" + test_name + "_" + str(idx) + "_"
                        + str(n_steps_taken) + "_capped_surface.vtp",
                    )
                    sitk.WriteImage(
                        capped_seg,
                        dir_output + "/" + case + "_" + test_name + "_"
                        + str(n_steps_taken) + "_capped_seg.mha",
                    )

        if global_config["PREVENT_RETRACE"]:
            if len(inside_pts) > 0:
                final_inside_pts = vf.appendPolyData(inside_pts)
                vf.write_vtk_polydata(
                    final_inside_pts,
                    dir_output + "/final_" + case + "_" + test_name + "_" + str(idx) + "_"
                    + str(n_steps_taken) + "_inside_points.vtp",
                )

        if not global_config["DEBUG"]:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
