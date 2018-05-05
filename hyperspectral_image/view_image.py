from spectral import imshow

class view_image():

    def __init__(self, image, fcc=True):
        """
        Ploats image, make sure image is loaded into memory using img.load_image() function.

        fcc = true loads (98,56,36) for false color composite

        """
        try:
            if fcc==False:
                imshow(image)
            else:
                imshow(image, (98,56,36))
        except:
            print( 'Error : Load image first and try again.' )
