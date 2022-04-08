# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:38:53 2021

@author: 33675
"""

class buildDataset(Dataset):
    # load the dataset definitions
    def buildDatasetLoading(self, dataset_dir, is_train=True):
        
        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "sac")
        self.add_class('dataset', 2, "cart")
        self.add_class('dataset', 3, "bout")
        
        print(dataset_dir + '/images/')
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            #image_id = filename[:-4]
            temp = re.findall("\d+", filename)
            test_is_train = re.split("\_", filename)[0] # On extrait le préfix du nom de l'image pour savoir si c'est train ou validation

            image_id = temp[0]
                        
            # skip bad images
            #if image_id in ['00090']:
               # continue
            # skip all images after 150 if we are building the train set
            if is_train and test_is_train == "val":
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and test_is_train == "train":
                continue
            
            # setting image file
            img_path = images_dir + filename
            
            # getting prefix images
            pref_img = re.split("\_", filename)[1]
            
            # setting annotations file
            ann_path = annotations_dir + pref_img + '_' + image_id + '.xml'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            
# extract bounding boxes from an annotation file
    def buildDatasetBoxes(self, filename):
        
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        classes = []
        lst_class = root.findall('.//name') # liste des classes présentes dans le xml

        box_id = 0
        
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            classes.append(self.class_names.index(lst_class[box_id].text))
            box_id = box_id + 1
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, classes, width, height
# load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, classes, w, h = self.buildDatasetBoxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            #class_ids.append(self.class_names.index('sac'))
        return masks, np.asarray(classes, dtype='int32')
# load an image reference
    def buildDatasetReference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']