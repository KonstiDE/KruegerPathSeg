import re
import requests

# hat-tip to chatgpt
def parse_requirements(file_path):
    packages = []
    with open(file_path, 'r') as f:
        for line in f:
            # Ignore comments and empty lines
            line = line.strip()
            if line and not line.startswith("#"):
                # Split on `==` to get package name and version
                package_match = re.match(r"([a-zA-Z0-9_\-]+)==([\d\.]+)", line)
                if package_match:
                    package, version = package_match.groups()
                    packages.append((package, version))
    return packages


def fetch_package_metadata(package, version):
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def generate_bibtex_entry(package_data):
    info = package_data['info']
    name = info.get('name')
    version = info.get('version')
    author = info.get('author', 'Unknown Author')
    summary = info.get('summary', 'No description available')
    url = info.get('project_url', f"https://pypi.org/project/{name}/{version}/")
    keywords = info.get('keywords', 'No keywords available')

    bibtex_entry = f"""@misc{{{name.lower()}_{version},
      title = {{{name}}},
      author = {{{author}}},
      year = {{{info.get('release_date', 'n.d.')}}},
      note = {{Version {version}. {summary}}},
      url = {{{url}}},
      keywords = {{{keywords}}},
    }}"""
    return bibtex_entry


def main(requirements_file='requirements.txt'):
    packages = parse_requirements(requirements_file)
    bibtex_entries = []

    for package, version in packages:
        package_data = fetch_package_metadata(package, version)
        if package_data:
            bibtex_entry = generate_bibtex_entry(package_data)
            bibtex_entries.append(bibtex_entry)
        else:
            print(f"Could not fetch data for {package}=={version}")

    with open("citations.bib", "w") as f:
        f.write("\n\n".join(bibtex_entries))
        print("BibTeX citations saved to citations.bib")


if __name__ == "__main__":
    main(requirements_file="C:/Users/s371513/Desktop/requirements.txt")
