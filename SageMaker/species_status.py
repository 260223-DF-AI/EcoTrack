import os

class SpeciesStatuses():
    def __init__(self, cautious: bool = True):
        """If cautious set to True, the worst status in endangered.txt will be assumed. False will assumed best status"""
        self.statuses = {
            "NE": "NOT EVALUATED", #no birds, using this for non-bird images
            "DD": "DATA DEFICIENT",
            "LC": "LEAST CONCERN",
            "NT": "NEAR THREATENED",
            "VU": "VULNERABLE",
            "EN": "ENDANGERED",
            "CR": "CRITICALLY ENDANGERED",
            "RE": "REGIONALLY EXTINCT",
            "EW": "EXTINCT IN THE WILD",
            "EX": "EXTINCT"
        }

        self.species = {}

        # for root, dirs, files in os.walk("animals"):
            # print(root)

        with open("./SageMaker/endangered.txt", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line[0] == "#" or line == '': continue
                animal = line.replace(',', '').replace('\n', '').split()
                label = animal[1]
                status = animal[2:]
                caution = -1 if cautious else 0
                self.species[idx] = (label, status, self.statuses[status[caution]])                
                # print(self.species[idx])

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
    print(species_statuses[0])