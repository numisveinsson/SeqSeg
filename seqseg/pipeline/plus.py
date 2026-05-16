"""Sweep + SeqSeg (seqseg_plus) batch pipeline."""

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
from seqseg.modules.datasets import get_testing_samples
from seqseg.modules.simvascular import write_simvascular_proj
from seqseg.modules.sweep import run_global_segmentation
from seqseg.modules.tracing import TracingContext, trace_centerline_from_context
from seqseg.pipeline.directories import create_case_directories


def run_plus_batch(
    *,
    dir_output0: str,
    data_dir: str,
    seqseg_dataset: str,
    dir_model_weights_seqseg: str,
    seqseg_fold: str,
    seqseg_test_name: str,
    seqseg_scale: float,
    dir_model_weights_global: str,
    global_fold: str,
    global_scale: float,
    img_format: str,
    unit: str,
    max_step_size: int,
    max_n_branches: int,
    max_n_steps_per_branch: int,
    write_samples: bool,
    take_time: bool,
    calc_global_centerline: bool,
    cap_surface_cent: bool,
    assembly_threshold: float,
    num_seeds: int,
    resample_spacing: Optional[Sequence[float]],
    global_config: Mapping[str, Any],
    start: int,
    stop: int,
    start_time_global: float,
) -> None:
    testing_samples, directory_data = get_testing_samples(seqseg_dataset, data_dir)

    for test_case in testing_samples[start:stop]:
        print(f"\n{test_case}\n")

        dir_output, dir_image, dir_seg, dir_cent, case, idx, json_file_present = (
            init.process_init(
                test_case, directory_data, dir_output0, img_format, seqseg_test_name
            )
        )

        if resample_spacing is not None:
            os.makedirs(dir_output, exist_ok=True)
            dir_image, dir_seg = sf.maybe_resample_volume_paths(
                dir_image,
                dir_seg,
                resample_spacing,
                dir_output,
                case,
                img_format,
            )

        pred_sweep, prob_pred_sweep = run_global_segmentation(
            dir_image=dir_image,
            model_folder=dir_model_weights_global,
            fold=global_fold,
            scale=global_scale,
        )
        sitk.WriteImage(pred_sweep, dir_output0 + "/" + case + "_sweep_seg.mha")
        sweep_surface = vf.evaluate_surface(pred_sweep, 1)
        vf.write_vtk_polydata(
            sweep_surface,
            dir_output0 + "/" + case + "_sweep_surface_nonsmooth.vtp",
        )
        sweep_surface_smooth = vf.taubin_smooth_polydata(
            sweep_surface, it=75, mu1=0.5, mu2=0.51
        )
        sweep_surface_smooth = vf.compute_polydata_normals(sweep_surface_smooth)
        vf.write_vtk_polydata(
            sweep_surface_smooth,
            dir_output0 + "/" + case + "_sweep_surface.vtp",
        )

        create_case_directories(dir_output, write_samples)
        vf.write_image_as_vti(
            dir_image,
            os.path.join(dir_output, "simvascular", "Images", f"{case}.vti"),
        )
        write_simvascular_proj(os.path.join(dir_output, "simvascular"))

        potential_branches, initial_seeds, sweep_centerline = init.initialize_from_seg(
            pred_sweep,
            dir_output,
            num_seeds=num_seeds,
            return_centerline=True,
        )
        if sweep_centerline is not None:
            vf.write_vtk_polydata(
                sweep_centerline,
                dir_output0 + "/" + case + "_sweep_centerline.vtp",
            )
        else:
            print("Sweep centerline extraction failed in initialization")

        ref_image = sitk.ReadImage(dir_image)
        sf.validate_potential_branches_image_bounds(
            ref_image, potential_branches, case=case, image_path=dir_image
        )

        if not global_config.get("DEBUG", False):
            sys.stdout = open(dir_output + "/out.txt", "w")
        else:
            print("Start tracing with debug mode on")

        print(test_case)
        print(f"Initial points: {potential_branches}")
        print(f"Time is: {time.time()}")

        ctx = TracingContext(
            output_folder=dir_output,
            image_file=dir_image,
            case=case,
            model_folder=dir_model_weights_seqseg,
            fold=seqseg_fold,
            potential_branches=potential_branches,
            max_step_size=max_step_size,
            max_n_branches=max_n_branches,
            max_n_steps_per_branch=max_n_steps_per_branch,
            global_config=global_config,
            unit=unit,
            scale=seqseg_scale,
            seg_file=dir_seg,
            start_seg=prob_pred_sweep,
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
            + str((time.time() - start_time_global) / 60) + " min\n"
        )

        if take_time:
            vessel_tree.time_analysis()

        if global_config.get("TREE_ANALYSIS", False):
            vessel_tree.create_tree_polydata_v1(dir_output)
            vessel_tree.create_tree_polydata_v2(dir_output)
            vessel_tree.plot_radius_distribution(dir_output)

        assembly_org = assembly_obj.assembly
        print(
            "\nTotal calculation time is:"
            + str((time.time() - start_time_global) / 60) + " min\n"
        )

        assembly = assembly_org

        assembly_binary = sitk.BinaryThreshold(
            assembly,
            lowerThreshold=assembly_threshold,
            upperThreshold=1,
        )
        sitk.WriteImage(
            assembly_binary,
            dir_output + "/" + case + "_raw_seg_" + seqseg_test_name + "_" + str(idx) + ".mha",
        )

        assembly_binary = sf.keep_component_seeds(assembly_binary, initial_seeds)
        sitk.WriteImage(
            assembly_binary,
            dir_output0 + "/" + case + "_segmentation_" + str(n_steps_taken) + "_steps.mha",
        )

        assembly_surface = vf.evaluate_surface(assembly_binary, 1)
        vf.write_vtk_polydata(
            assembly_surface,
            dir_output + "/" + case + "_surface_" + str(n_steps_taken) + "_steps.vtp",
        )
        surface_smooth = vf.smooth_polydata(assembly_surface)
        vf.write_vtk_polydata(
            surface_smooth,
            dir_output0 + "/" + case + "_surface_" + str(n_steps_taken) + "_steps.vtp",
        )

        final_surface = vf.appendPolyData(surfaces)
        final_centerline = vf.appendPolyData(centerlines)
        final_points = vf.appendPolyData(points)

        vf.write_vtk_polydata(
            final_surface,
            dir_output + "/all_" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
            + str(n_steps_taken) + "_surfaces.vtp",
        )
        vf.write_vtk_polydata(
            final_centerline,
            dir_output + "/all_" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
            + str(n_steps_taken) + "_centerlines.vtp",
        )
        vf.write_vtk_polydata(
            final_points,
            dir_output + "/all_" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
            + str(n_steps_taken) + "_points.vtp",
        )

        if calc_global_centerline:
            global_centerline, targets, success = calc_centerline_global(
                assembly_binary,
                initial_seeds,
                nr_seeds=num_seeds,
                merge_method=global_config.get("CENTERLINE_MERGE_METHOD", "clean"),
            )
            if success:
                vf.write_vtk_polydata(
                    global_centerline,
                    dir_output0 + "/" + case + "_centerline_" + str(n_steps_taken) + "_steps.vtp",
                )
                targets_pd = vf.points2polydata(
                    [target.tolist() for target in targets]
                )
                vf.write_vtk_polydata(
                    targets_pd,
                    dir_output + "/" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
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
                        dir_output + "/" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
                        + str(n_steps_taken) + "_capped_surface.vtp",
                    )
                    sitk.WriteImage(
                        capped_seg,
                        dir_output + "/" + case + "_" + str(n_steps_taken) + "_capped_seg.mha",
                    )

        if global_config.get("PREVENT_RETRACE", False):
            final_inside_pts = vf.appendPolyData(inside_pts)
            vf.write_vtk_polydata(
                final_inside_pts,
                dir_output + "/final_" + case + "_" + seqseg_test_name + "_" + str(idx) + "_"
                + str(n_steps_taken) + "_inside_points.vtp",
            )

        if not global_config.get("DEBUG", False):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
