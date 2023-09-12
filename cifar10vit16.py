import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import random
from tqdm import tqdm


# helper functions to get the data in a form I can use for inference
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data():
    label_names = unpickle('../../../../groups/course.cap6411/cifar-10-batches-py/batches.meta')
    labels = []
    for i in range(len(label_names[b'label_names'])):
        # strip the b' and ' from the label names
        labels.append(label_names[b'label_names'][i].decode("utf-8"))


    # load the image data
    data = unpickle('../../../../groups/course.cap6411/cifar-10-batches-py/data_batch_1')
    images_data = data[b'data']
    targets = data[b'labels']

    new_images_data = []

    for img in images_data:
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img, 'RGB')
        new_images_data.append(img)

    return new_images_data, targets, labels


# step 1: grab a pretrained model
model_name = "nateraw/vit-base-patch16-224-cifar10"
model = ViTForImageClassification.from_pretrained(model_name)

# step 2: get the cifar 10 data from our helper function
images_data, targets, label_array = get_data()
# this is for the video demo, just grab a single random image
test = random.randint(0, len(images_data))
image = images_data[test]

# step 3: Use the ViT feature extractor to preprocess the image
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
inputs = feature_extractor(images=image, return_tensors="pt")

# Step 4: Run Inference
outputs = model(**inputs)
logits = outputs.logits  # Get the logits (raw scores) from the model
probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

# print the probabilities
for label in range(len(label_array)):
    print(f"{label_array[label]}: {probabilities[label].item() * 100:.2f}%")

# Find the predicted class index with the highest probability
predicted_class = torch.argmax(probabilities).item()
predicted_label = label_array[predicted_class]

print(f"Predicted label: {predicted_label}")
print(f"Ground truth label: {label_array[targets[test]]}")

print(f"are they the same? {predicted_label == label_array[targets[test]]}")

run_inference = True
if run_inference:
    # step 5: run inference on all the images
    correct = 0
    for i in tqdm(range(len(images_data)), desc="current image"):
        image = images_data[i]
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits  # Get the logits (raw scores) from the model
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Find the predicted class index with the highest probability
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = label_array[predicted_class]

        if predicted_label == label_array[targets[i]]:
            correct += 1

    print(f"Accuracy: {correct / len(images_data) * 100:.2f}%")


# show the image
image.show()
