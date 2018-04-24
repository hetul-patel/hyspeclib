import psutil

class check_memory:

    def check_image_before_load(self,image_dims):
        """
        Check if image can be fit into memory or not without loading it.

        """

        if image_dims[0]*image_dims[1]*image_dims[2]*4 < self.check_available_memory():
            return True
        else:
            return False

    def check_available_memory(self,unit='B'):
        """
        Returns available memory in RAM in desired unit.

        unit : 'MB' for mega bytes, 'GB' for giga bytes, default bytes.

        """
        free = psutil.virtual_memory().available

        if unit == 'MB':

            return free/10**6

        elif unit == 'GB':

            return free/10**9

        else:

            return free