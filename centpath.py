import vtk
import os
import numpy as np
import xml.etree.ElementTree as ET


class ImageProcessing:
    def __init__(self, centerlines, output_directory):
        self.centerlines = centerlines
        self.output_directory = output_directory
        self.paths = []
        
    def extract_paths(self, dist_mult, tangent_change_deg):
        if self.centerlines is None:
            print("No centerlines have been created.")
            return
        
        image_spacing = self.get_image_spacing()
        dist_measure = max(image_spacing)
        
        sections = self.extract_centerline_sections()
        self.clear_old_path_files()
        
        path_id = 1
        for section in sections:
            path_points = self.sample_line_points(section, dist_mult, tangent_change_deg, dist_measure)
            
            path_elem = {
                "method": "CONSTANT_TOTAL_NUMBER",
                "calculation_number": 100,
                "control_points": path_points
            }
            
            self.write_path_file(path_id, path_elem)
            self.paths.append(path_elem)
            path_id += 1
        
        print("Paths extraction complete.")
        
    def get_image_spacing(self):
        return [1.0, 1.0, 1.0]  # Placeholder function
    
    def extract_centerline_sections(self):
        sections = []
        centerline_ids = self.centerlines.GetCellData().GetArray("CenterlineIds")
        if centerline_ids is None:
            raise RuntimeError("No 'CenterlineIds' cell data array found in centerlines geometry.")
        
        vrange = centerline_ids.GetRange()
        min_id, max_id = int(vrange[0]), int(vrange[1])
        
        for cid in range(min_id, max_id + 1):
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(self.centerlines)
            threshold.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "CenterlineIds")
            threshold.ThresholdBetween(cid, cid)
            threshold.Update()
            
            surfacer = vtk.vtkDataSetSurfaceFilter()
            surfacer.SetInputData(threshold.GetOutput())
            surfacer.Update()
            
            centerlines_cid_threshold = surfacer.GetOutput()
            
            group_data = centerlines_cid_threshold.GetCellData().GetArray("GroupIds")
            if group_data:
                lower_value, upper_value = group_data.GetRange()
                group_threshold = vtk.vtkThreshold()
                group_threshold.SetInputData(centerlines_cid_threshold)
                group_threshold.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "GroupIds")
                group_threshold.ThresholdBetween(lower_value, upper_value)
                group_threshold.Update()
                
                group_surfacer = vtk.vtkDataSetSurfaceFilter()
                group_surfacer.SetInputData(group_threshold.GetOutput())
                group_surfacer.Update()
                
                group_centerlines = vtk.vtkPolyData()
                group_centerlines.DeepCopy(group_surfacer.GetOutput())
                sections.append(group_centerlines)
            else:
                sections.append(centerlines_cid_threshold)
        
        return sections
    
    def sample_line_points(self, section, dist_mult, tangent_change_deg, dist_measure):
        points = section.GetPoints()
        num_points = section.GetNumberOfPoints()
        
        if num_points < 2:
            return []
        
        clength = 0.0
        pt1, pt2 = [0, 0, 0], [0, 0, 0]
        for i in range(num_points - 1):
            points.GetPoint(i, pt1)
            points.GetPoint(i + 1, pt2)
            clength += np.linalg.norm(np.array(pt2) - np.array(pt1))
        
        max_dist = dist_mult * dist_measure
        num_samples = clength / max_dist
        if num_samples < 1.0:
            max_dist = clength / 2.0
        
        sample_points = []
        last_point, last_tangent = None, None
        
        for i in range(num_points - 1):
            points.GetPoint(i, pt1)
            points.GetPoint(i + 1, pt2)
            tangent = np.array(pt2) - np.array(pt1)
            tangent /= np.linalg.norm(tangent)
            
            add_point = False
            if i == 0:
                last_tangent = tangent
                last_point = np.array(pt1)
                add_point = True
            else:
                if np.dot(last_tangent, tangent) < np.cos(np.radians(tangent_change_deg)):
                    add_point = True
                elif np.linalg.norm(np.array(pt1) - last_point) > max_dist:
                    add_point = True
            
            if add_point:
                last_tangent = tangent
                last_point = np.array(pt1)
                sample_points.append(list(pt1))
        
        points.GetPoint(num_points - 1, pt1)
        sample_points.append(list(pt1))
        
        return sample_points

    def clear_old_path_files(self):
        for filename in os.listdir(self.output_directory):
            if filename.endswith(".pth"):
                os.remove(os.path.join(self.output_directory, filename))
    
    def write_path_file(self, path_id, path_elem):
        file_path = os.path.join(self.output_directory, f"path_{path_id}.pth")
        
        root = ET.Element("format")
        root.set("version", "1.0")
        
        path_element = ET.SubElement(root, "path")
        path_element.set("id", str(path_id))
        path_element.set("method", path_elem["method"])
        path_element.set("calculation_number", str(path_elem["calculation_number"]))
        path_element.set("spacing", str(self.get_image_spacing()))
        
        timestep_element = ET.SubElement(path_element, "timestep")
        timestep_element.set("id", "0")
        
        path_points_element = ET.SubElement(timestep_element, "path_points")
        for i, point in enumerate(path_elem["control_points"]):
            path_point_element = ET.SubElement(path_points_element, "path_point")
            path_point_element.set("id", str(i))
            pos_element = ET.SubElement(path_point_element, "pos", x=str(point[0]), y=str(point[1]), z=str(point[2]))
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        
        print(f"Path {path_id} saved to {file_path}.")
