"""#-----------------------------TALLER 3: filtering-------------------------------------------------------------------
                              Erick Steven Badillo Vargas
                                Ingenieria Electronica
                           Procesamiento de imagenes y vision
                                    Julian Quiroga
                                         2020
#----------------------------------------------------------------------------------------------------------------------#"""

#Imports
import cv2
import numpy as np

class filter:

    def filter_gauss(self,image):					# Filtro Gaussiano
        sigma= 1.5                              			# Derivacion estandar
        N=7                                     			# Ventana
        f_image_gauss=cv2.GaussianBlur(image, (N,N),sigma) 	# Filtro gaus,ventana NxN y sigma
        return f_image_gauss						# Retorna el la imagen filtrada con el filtro de Gauss

    def filter_median(self,image):					# Filtro Mediana
        f_image_median= cv2.medianBlur(image, 7)	# Filtro mediana de 7x7
        return f_image_median						# Retorna la imagen filtrada con el filtro mediana

    def filter_bilateral(self,image):					            # Filtro bilateral
        image_filtered=cv2.bilateralFilter(image, 15, 25, 25)		# Filtro bilateral con d, sigmacolor y sigmaspace
        return image_filtered						                # Retorna imagen filtrada con el filtro bilateral

    def filter_nlm(self,image):						# Filtro de promedios no locales
        image_filterednml=cv2.fastNlMeansDenoising(image, 5, 15, 25)	# Filtro nlm con h, windowsize y searchsize
        return image_filterednml					# Retorna imagen filtrada con el filtro de promedios no lineales

    def rmse(self,image_gray, image_filtered):				# Error cuadratico medio
        M=image_gray.shape[0]                               # M (alto de la imagen)
        N=image_gray.shape[1]                               # N (ancho de la imagen)
        ECM =(np.square(image_gray - image_filtered)).mean()  # calculo del ECM
        return np.sqrt(ECM)


