import xml.etree.ElementTree as ET


def transform():
    model_path = 'C:/Users/s371513/Desktop/master_qgis/models/calculate_vertecies.model3'
    tree = ET.parse(model_path)
    root = tree.getroot()

    dot_output_path = 'C:/Users/s371513/Desktop/master_qgis/models/calculate_vertecies.dot'

    processes = {}
    connections = []

    for process in root.findall(".//Option[@type='Map']"):
        try:
            process_id = process.find("./Option[@name='id']").get('value')
            alg_id = process.find("./Option[@name='component_description']").get('value')

            if process_id and alg_id:
                processes[process_id] = alg_id

            for param in [process.find(".//Option[@name='INPUT']"), process.find(".//Option[@name='input']")]:
                if param is not None:
                    child_id = param.find(".//Option[@name='child_id']").get('value')
                    if child_id:
                        connections.append((child_id, process_id))

        except (AttributeError, TypeError) as _:
            continue

    with open(dot_output_path, 'w+') as dot_file:
        dot_file.write('digraph QGIS_Model {\n')

        # Add nodes
        for process_id, alg_id in processes.items():
            dot_file.write(f'  "{process_id}" [label="{alg_id}"];\n')

        for source, target in connections:
            dot_file.write(f'  "{source}" -> "{target}";\n')

        dot_file.write('}\n')


if __name__ == '__main__':
    transform()
