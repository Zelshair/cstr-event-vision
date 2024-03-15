from torchvision import models
import torch
from config import Configs

class ClassificationModel(torch.nn.Module):
    def __init__(self, num_classes, channels, width, height, classifier='resnet50', pretrained=False):
        super().__init__()

        # set number of channels
        self.channels = channels

        # set input height and width
        self.height = height
        self.width = width

        # set output number of classes
        self.num_classes = num_classes

        self.pretrained = pretrained

        # create custom classifier with desired classifier, then move to preferred device
        self.classifier = create_classifier(classifier, self.channels, self.num_classes, pretrained=self.pretrained).to(Configs().DEVICE)

    def forward(self, input):
        out = self.classifier(input)
        return out

def create_classifier(classifier, channels, num_classes, pretrained=False):
    '''
    This function creates a classifier using a defined torchvision.models classification network and 
    updates the input layer and output layers to support the desired number of input channels and output classes.
    
    Currently supported classifiers: {'resnet18', 'resnet50', 'inceptionv3', 'mobilenetv2', 'mobilenetv3l', 'mobilenetv3s', 'densenet121', 'vgg16', 'vgg19', 'vit_b_32', 'vit_b_16'}'''

    if classifier == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)    

    elif classifier == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        
        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)    

    elif classifier == 'inceptionv3':
        if pretrained:
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        else:
            model = models.inception_v3(weights=None)
        model.aux_logits = False

        assert channels==3, "InceptionV3 only accepts a 3-channel input with a resolution of 299x299. A {}-channel input is not supported".format(channels)

        # if channels != 3:
        #     # Replace the input layer to match the number of input channels (If not set to 3)
        #     model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)    
    
    # InceptionV3 with the Aux network
    elif classifier == 'inceptionv3_aux':
        if pretrained:
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        else:
            model = models.inception_v3(weights=None)
        
        model.aux_logits = True

        assert channels==3, "InceptionV3 only accepts a 3-channel input with a resolution of 299x299. A {}-channel input is not supported".format(channels)

        # if channels != 3:
        #     # Replace the input layer to match the number of input channels (If not set to 3)
        #     model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)    
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes) # Handle the primary net 

    elif classifier == 'mobilenetv2':
        if pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            model = models.mobilenet_v2(weights=None)

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0][0] = torch.nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)    

    elif classifier == 'mobilenetv3l':
        if pretrained:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = models.mobilenet_v3_large(weights=None)
        
        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0][0] = torch.nn.Conv2d(channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes) 

    elif classifier == 'mobilenetv3s':
        if pretrained:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            model = models.mobilenet_v3_small(weights=None)

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0][0] = torch.nn.Conv2d(channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes) 

    elif classifier == 'densenet121':
        if pretrained:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            model = models.densenet121(weights=None)
        
        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0] = torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the fully connected layer to match number of output classes
        model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)            

    elif classifier == 'vgg16':
        if pretrained:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model = models.vgg16(weights=None) 

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0] = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # update the output: replace the last layer of the fully connected network to match number of classes
        model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)

    elif classifier == 'vgg19':
        if pretrained:
            model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            model = models.vgg19(weights=None) 

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.features[0] = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # update the output: replace the last layer of the fully connected network to match number of classes
        model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)  

    elif classifier == 'vit_b_32':
        if pretrained:
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT) 

        else:
            model = models.vit_b_32(weights=None)

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.conv_proj = torch.nn.Conv2d(channels, 768, kernel_size=(32, 32), stride=(32, 32), padding=(0, 0))

        # update the output: replace the last layer of the fully connected network to match number of classes
        model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes)

    elif classifier == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) 
            # best weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 but requires a min size of 384x384
            # 2nd best weights = IMAGENET1K_SWAG_LINEAR_V1 (min size 224x224)
            # default is ViT_B_16_Weights.IMAGENET1K_V1
        else:
            model = models.vit_b_16(weights=None)

        if channels != 3:
            # Replace the input layer to match the number of input channels (If not set to 3)
            model.conv_proj = torch.nn.Conv2d(channels, 768, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0))

        # update the output: replace the last layer of the fully connected network to match number of classes
        model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes)

    else:
        TypeError("Classification model '{}' is not supported!".format(classifier))

    return model