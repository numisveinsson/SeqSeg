import vtk
import xml.etree.ElementTree as ET

def read_vtp_file(file_path):
    """
    Reads a VTP file and extracts point coordinates.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()

    points = polydata.GetPoints()
    point_coordinates = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]

    return point_coordinates

def pretty_print_element(element, level=0, indent="    "):
    """
    Recursively formats an XML element for pretty printing.
    """
    spacing = "\n" + level * indent
    if len(element):  # If the element has children
        element.text = spacing + indent
        for child in element:
            pretty_print_element(child, level + 1, indent)
        child.tail = spacing
    if level and not element.tail:
        element.tail = spacing
    elif not level:
        element.tail = "\n"

def write_pth_file(file_path, point_coordinates):
    """
    Writes point data into a PTH file in the required format.
    """
    root = ET.Element("format", version="1.0")
    path = ET.SubElement(root, "path", id="104", method="0", calculation_number="100", spacing="0")
    timestep = ET.SubElement(path, "timestep", id="0")
    path_element = ET.SubElement(timestep, "path_element", method="0", calculation_number="100", spacing="0")

    # Add control points
    control_points = ET.SubElement(path_element, "control_points")
    for i, (x, y, z) in enumerate(point_coordinates):
        ET.SubElement(control_points, "point", id=str(i), x=f"{x}", y=f"{y}", z=f"{z}")

    # Add path points with example tangents and rotations (placeholders for now)
    path_points = ET.SubElement(path_element, "path_points")
    for i, (x, y, z) in enumerate(point_coordinates):
        path_point = ET.SubElement(path_points, "path_point", id=str(i))
        ET.SubElement(path_point, "pos", x=f"{x}", y=f"{y}", z=f"{z}")
        ET.SubElement(path_point, "tangent", x="0", y="0", z="1")  # Example tangent
        ET.SubElement(path_point, "rotation", x="0", y="1", z="0")  # Example rotation

    # Pretty print the XML
    pretty_print_element(root)

    # Write to file
    tree = ET.ElementTree(root)
    with open(file_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)

def main():
    vtp_file = "/Users/numisveins/Documents/datasets/vascular_data_3d/centerlines/0160_6001.vtp"  # Input VTP file
    pth_file = "output.pth"  # Output PTH file

    # Read the VTP file
    print(f"Reading VTP file: {vtp_file}")
    point_coordinates = read_vtp_file(vtp_file)

    # Write the PTH file
    print(f"Writing PTH file: {pth_file}")
    write_pth_file(pth_file, point_coordinates)

    print("Conversion complete.")

if __name__ == "__main__":
    main()
