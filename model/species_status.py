

class SpeciesStatuses():
    def __init__(self):
        self.statuses = {
            "NE": "NOT EVALUATED", #no birds, using this for non-bird images
            "DD": "DATA DEFICIENT",
            "LC": "LEAST CONCERN",
            "NT": "NEAR THREATENED",
            "VU": "VULNERABLE",
            "EN": "ENDANGERED",
            "CR": "CRITICALLY ENDANGERED",
            "EW": "EXTINCT IN THE WILD",
            "EX": "EXTINCT"
        }

        self.species = {}

        with open("./model/endangered.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[0] == "#": continue
                _, folder, status = line.replace(',', '').replace('\n', '').split()
                label, species = folder.replace('_', ' ').split('.')
                label = int(label)
                multi = '*' in status
                status = status.replace('*', '')
                self.species[label] = (species, self.statuses[status], multi)
                # print(label, species, status, multi)
                

    def __getitem__(self, label: int):
        """
        Returns tuple of species name, status, and whether there are 
        multiple varied statuses within the species (return status is 
        of the species of most concern, so if two subspecies are 
        endanged and least concern, the species will be listed as 
        endangered).

        Labels are 1-indexed, so when using with our model that outputs
        0-indexed classes, they should be fed to getitem with +1
        """
        if len(self.species) < label:
            return None
        return self.species[label]
    
if __name__ == "__main__":
    species_statuses = SpeciesStatuses()
    print(species_statuses[10])