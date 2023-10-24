from glob import glob
from torchvision.datasets import CIFAR10
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

def main(args):
    all_paths  = []
    for path in glob('./cifar10/*/*/*/*00*'):
        all_paths.append(path)
    g = all_paths[args.id]

    f = open(str(g), "r")
    count = 0
    index, label, prediction, radius, correct = [], [], [], [], []
    for line in f:
        if count > 0:
            idx, lab, pre, rad, cor, _, _ = line.split('\t')
            radius.append(float(rad))
            correct.append(int(cor))
            prediction.append(int(pre))
            label.append(int(lab))
            index.append(int(idx))
            
        else:
            count += 1

    dataset = CIFAR10(root='./train/datasets', train=False, download=True, transform=ToTensor())

    saved_images, saved_predictions, saved_radii = [], [], []
    anything_detected = False
    for i in tqdm(range(len(radius))):
        
        idx, pred, rad = index[i], prediction[i], radius[i]
        img, _ = dataset[idx]

        if saved_images == []:
            saved_images.append(img)
            saved_predictions.append(pred)
            saved_radii.append(rad)
        
        else:
            #Get the differences
            diff = torch.norm(img.reshape(1, -1) - torch.stack(saved_images).reshape(len(saved_radii), -1), dim=1)
            
            where_overlap =  diff < (torch.tensor(saved_radii) + rad)
            #Check whether this image is with overlap with any other instances
            if where_overlap.any():
                print("There is overlap")
                preds_overlap = torch.tensor(saved_predictions)[where_overlap]
                where_overlap_diff_class = preds_overlap != pred
                
                #Check whether this image is with overlap with instances with different prediction
                if where_overlap_diff_class.any():
                    print("There is overlap with different class")
                    anything_detected=True # We will save something
                    #Get the radii, differences where the overlap
                    saved_radii_with_overlap = torch.tensor(saved_radii)[where_overlap]
                    dif_with_overlap = diff[where_overlap]

                    preds_overlap_with_diff_class = preds_overlap[where_overlap_diff_class]
                    rad_with_overlap_diff_class = saved_radii_with_overlap[where_overlap_diff_class]
                    dif_with_overlap_diff_class = dif_with_overlap[where_overlap_diff_class]

                    rad, rad_idx = torch.min(dif_with_overlap_diff_class - rad_with_overlap_diff_class)
                    
                    if rad.item() < 0:
                        assert preds_overlap_with_diff_class[rad_idx] != pred, 'No way!'
                        pred = preds_overlap_with_diff_class[rad_idx]

                    rad = torch.abs(rad).item()

            
            saved_images.append(img)
            saved_predictions.append(pred)
            saved_radii.append(rad)

    print("You are Done!")

    if anything_detected:
        f = open('./results/' + str(args.id) + '.txt', 'w')
        print(g, file=f, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=int, default=128, help="id of the path")
    args = parser.parse_args()
    main(args)
