import os
from PIL import Image
from torchvision import transforms
from SMLocal import upload, deploy, predict, shutdown
from species_status import SpeciesStatuses

def main():
    upload()

    species_statuses = SpeciesStatuses()

    std_transform = transforms.Compose([
    	transforms.Resize((256, 256)),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize( #these are provided hardcoded values
    		mean=[0.485, 0.456, 0.406],
    		std=[0.229, 0.224, 0.225]
    	) 
    ])

    predictor = deploy()

    try:
        while True:
            img_path = input("Enter the path to your image: ")
            if img_path.lower() in ['n', 'no', 'exit', 'q', 'quit']:
                break
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                print("Image does not exist at that location")
                exit(1)

            input_data = std_transform(img).unsqueeze(0)
            predictor, confidence, label = predict(predictor, input_data)
            print(f"Class: {species_statuses[label]}, %{confidence:.2f} confidence.")
    except Exception as e:
        print(e)
    finally:
        shutdown(predictor)

if __name__ == "__main__":
    main()