import os
from imports import *

file_path = 'data/'
df = pd.read_csv(file_path+'all_data_info.csv')
df.head()

archive = zipfile.ZipFile(file_path+'replacements_for_corrupted_files.zip', 'r')
corrupted_ids = set()

for item in archive.namelist():
    ID = re.sub("[^0-9]", "", item)
    if ID != "":
        corrupted_ids.add(ID)

drop_idx = []
for index, row in df.iterrows():
    id_check = re.sub("[^0-9]", "", row['new_filename'])
    if id_check in corrupted_ids:
        drop_idx.append(index)
df = df.drop(drop_idx)
artisti_numero = 15
paintings = df['artist'].value_counts().head(artisti_numero)
artists = paintings.index.tolist()
sample_size = min(paintings)


parser = argparse.ArgumentParser(description='Prepare dataset for training.')
parser.add_argument('--path', metavar='p', type=str, default=file_path,
                    help='Path to the data folder.')
parser.add_argument('--sample_size', metavar='s', type=int, default=sample_size,
                    help='Number of samples per artist.')
parser.add_argument('--artists', metavar='a', type=int, default=artisti_numero,
                    help='Number of artists to use.')
parser.add_argument('--img_size', metavar='i', type=int, default=224,
                    help='Size of the images.')

args = parser.parse_args()

file_path = args.path
sample_size = args.sample_size
paintings = df['artist'].value_counts().head(args.artists)
img_size = args.img_size

if file_path.endswith('data/'):
    paintings = df['artist'].value_counts().head(1)
    artists = paintings.index.tolist()
    sample_size = 3
    

ImageFile.LOAD_TRUNCATED_IMAGES=True


active_df = pd.DataFrame({}) # Reduce the large dataframe into the one containing only relevant data

for artist in artists:


    tr_df = df[(df['artist']==artist)].sort_values(by=['in_train','size_bytes'], ascending=[False, True])
    active_df = pd.concat([active_df,tr_df.iloc[:sample_size]])


# Label Encoder to transform artist names into integers from 0 to 19
LabEnc = preprocessing.LabelEncoder()
LabEnc.fit(artists)

matplotlib.rc_file_defaults()

def image_transformer_nn(image, new_dim=224):
    """
    Args:
        image: Image to transform
        new_dim: Dimension (pixels) to resize image
    """
    # Convert PIL image to tensor
    tensoring = transforms.ToTensor()
    image = tensoring(image)
    # Get the image's shape
    channels, height, width = image.shape
    # If width is smaller than height, we transpose the image
    if width < height:
        image = image.transpose(1,2)
    # Calculate the ratio between the image's original width and the new width
    res_percent = float(new_dim/width)
    # Calculate the new height
    height = round(height*res_percent)
    # Resize the image
    resizer = transforms.Resize((height,new_dim))
    image = resizer(image)
    # Now that the image is resized by keeping aspect ratio, we pad "down"
    padder = transforms.Pad([0,0,0,int(new_dim-height)])
    image = padder(image)
    # If channels is different than 3, repeat until it is
    if channels != 3:
        image = image.repeat(3,1,1)
    return image

class ImageDataset(Dataset):
    def __init__(self, path, dataframe, lab_encoder, img_size=224):
        """
        Args:
            dataframe: dataframe to use for the IDs
            lab_encoder: label encoder to transform artist names into integers
            img_size: size to be used
        """
        self.encoder = lab_encoder
        self.img_size = img_size
        self.feats, self.labels = self.get_all_items(path,dataframe)
        
    def get_all_items(self,path,dataframe):

        # We begin with the train.zip
        curr_df = dataframe[dataframe['in_train']==True]
        archive = zipfile.ZipFile(path+'train.zip', 'r')
        img_path = 'train/'

        feats = []
        labels = []

        for index, row in curr_df.iterrows():
            # Features
            file = row['new_filename']
            imgdata = archive.open(img_path+file)
            try:
                image = Image.open(imgdata)
                datum = image_transformer_nn(image, new_dim=self.img_size)
                
                print("Loaded", len(feats) ,"out of", len(curr_df), "images.")
                feats.append(datum)

                # Label
                artist = row['artist']
                label = self.encoder.transform([artist])[0]
                labels.append(label)
            except Image.DecompressionBombError:
                print(f"Skipped loading image {file} to avoid a DecompressionBombError.")

        # Same for the test.zip
        curr_df = dataframe[dataframe['in_train']==False]
        archive = zipfile.ZipFile(path+'test.zip', 'r')
        img_path = 'test/'

        for index, row in curr_df.iterrows():
            # Features
            file = row['new_filename']
            imgdata = archive.open(img_path+file)
            try:
                image = Image.open(imgdata)
                datum = image_transformer_nn(image, new_dim=self.img_size)
                print("Loaded", len(feats) ,"out of", len(curr_df), "images.")
                
                feats.append(datum)

                # Label
                artist = row['artist']
                label = self.encoder.transform([artist])[0]
                labels.append(label)
            except Image.DecompressionBombError:
                print(f"Skipped loading image {file} to avoid a DecompressionBombError.")

        feats = torch.stack(feats)
        labels = torch.LongTensor(labels)
        return feats, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item]
        
nn_data = ImageDataset(file_path, active_df, LabEnc, img_size=224)

# This cell splits data into train-val-test and creates DataLoaders in the case of NNs
def DataSplitter(data, ratios=[60,20,20], batches=None, shuffle=True, seed=None):
    """
    Args:
        data: dataset to be loaded into loaders
        batches: batch size for loaders
        ratios: list of integers, containing the ratios [train,val,test] for splitting
        shuffle: option to shuffle data
        seed: seed for shuffling
    """
    first_ratio = (ratios[1]+ratios[2])/sum(ratios)
    second_ratio = ratios[2]/(ratios[1]+ratios[2])
    
    if isinstance(data,ImageDataset): 
        
        labels = data.labels.numpy()
        
        train_indices, rest_indices = train_test_split(np.arange(len(labels)),
                                               test_size=first_ratio, shuffle=shuffle,
                                               random_state=seed, stratify=labels)
        
        rest_labels = data[rest_indices][1]
        
        val_indices, test_indices = train_test_split(rest_indices,
                                            test_size=second_ratio, shuffle=shuffle,
                                            random_state=seed, stratify=rest_labels)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(data,batch_size=batches,sampler=train_sampler)
        val_loader = DataLoader(data,batch_size=batches,sampler=val_sampler)
        test_loader = DataLoader(data,batch_size=batches,sampler=test_sampler)
        
        return train_loader, val_loader, test_loader
        
    else:
        print('Invalid data Type.')
        return