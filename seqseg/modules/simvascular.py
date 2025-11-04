import xml.etree.ElementTree as ET
import numpy as np


def compute_tangents(points):
    """Compute approximate tangent vectors using central differences."""
    tangents = []
    n = len(points)
    for i in range(n):
        if i == 0:
            t = np.array(points[1]) - np.array(points[0])
        elif i == n - 1:
            t = np.array(points[-1]) - np.array(points[-2])
        else:
            t = np.array(points[i + 1]) - np.array(points[i - 1])
        t_norm = np.linalg.norm(t)
        t = t / t_norm if t_norm > 0 else np.zeros(3)
        tangents.append(t)
    return tangents


def indent(elem, level=0):
    """Manual pretty-printer for XML (for Python < 3.9)."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_pth(points, output_path):
    """Create a SimVascular-style .pth XML file given a list of (x,y,z) points."""
    path_elem = ET.Element(
        "path",
        attrib={
            "id": "1",
            "method": "0",
            "calculation_number": "100",
            "spacing": "0",
            "version": "1.0",
            "reslice_size": "5",
            "point_2D_display_size": "",
            "point_size": "",
        },
    )

    timestep_elem = ET.SubElement(path_elem, "timestep", {"id": "0"})
    path_element_elem = ET.SubElement(
        timestep_elem,
        "path_element",
        {"method": "0", "calculation_number": "100", "spacing": "0"},
    )

    # --- Control points ---
    control_points_elem = ET.SubElement(path_element_elem, "control_points")
    for i, (x, y, z) in enumerate(points):
        ET.SubElement(
            control_points_elem,
            "point",
            {"id": str(i), "x": f"{x}", "y": f"{y}", "z": f"{z}"},
        )

    # --- Path points ---
    path_points_elem = ET.SubElement(path_element_elem, "path_points")
    tangents = compute_tangents(points)

    for i, ((x, y, z), t) in enumerate(zip(points, tangents)):
        path_point_elem = ET.SubElement(path_points_elem, "path_point", {"id": str(i)})

        # Position
        ET.SubElement(path_point_elem, "pos", {"x": f"{x}", "y": f"{y}", "z": f"{z}"})

        # Tangent
        ET.SubElement(
            path_point_elem,
            "tangent",
            {"x": f"{t[0]}", "y": f"{t[1]}", "z": f"{t[2]}"},
        )

        # Placeholder rotation (SimVascular will recompute)
        ET.SubElement(
            path_point_elem,
            "rotation",
            {"x": "0", "y": "1.0", "z": "0"},
        )

    # --- Write to file ---
    tree = ET.ElementTree(path_elem)
    try:
        # Python 3.9+ only
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        indent(path_elem)

    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"✅ SimVascular .pth file written to {output_path}")


if __name__ == "__main__":
    # Example usage
    example_points = [
        (-2.6108, 12.7837, 166.0114),
        (-2.5841, 12.7916, 165.9611),
        (-2.5053, 12.8150, 165.8120),
        (-2.3787, 12.8530, 165.5696),
        (-2.2108, 12.9046, 165.2429),
        (-2.0102, 12.9681, 164.8432),
    ]

    create_pth(example_points, "example_path.pth")
