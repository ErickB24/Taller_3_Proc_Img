"""#-----------------------------TALLER 3: filtering-------------------------------------------------------------------
                              Erick Steven Badillo Vargas
                                Ingenieria Electronica
                           Procesamiento de imagenes y vision
                                    Julian Quiroga
                                         2020
#----------------------------------------------------------------------------------------------------------------------#"""


# Imports
from time import time
import os
import cv2
import numpy as np
from noise import noise as ns
from filtering import filter


#ABRIR IMAGEN
print("Indique la direccion de la Imagen:")     # Imprimir en pantalla
path=input()                                    #direccion, ejemplo:
                                                # C:\Users\Erick\Desktop\OCTAVO SEMESTRE\PROC DE IMAGENES\lena.png
path_file = os.path.join(path)                  # Abrir archivo
imagen = cv2.imread(path_file)                  # Leer imagen
cv2.imshow('imagen',imagen) 	                # visualizar la imagen
cv2.waitKey(0)

#Imagen en Grises
gray= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)              # Imagen en grises
image_gray= gray.astype(np.float) / 255                     # normalizar imagen
cv2.imshow('imagen',image_gray) 	                # visualizar la imagen
cv2.waitKey(0)

#Ruido Gauss
image_gauss=ns('gauss',image_gray)                          # Ruido Gauss
image_gauss=(255 * image_gauss).astype(np.uint8)            # normalizar imagen a formato
cv2.imshow('imagen con ruido gauss',image_gauss) 	        # visualizar la imagen
cv2.waitKey(0)

#Ruido Sal y pimienta
image_syp=ns('s&p',image_gray)                              # Ruido sal y pimienta
image_syp=(255 * image_syp).astype(np.uint8)                # normalizar imagen a formato
cv2.imshow('imagen con ruido S&P',image_syp) 	            # visualizar la imagen
cv2.waitKey(0)

#-----------------------------------------------------------------------------------------------------------------------
image_gray=(255 * image_gray).astype(np.uint8) # volver a sus valores anteriores la imagen
#-----------------------------------------------------------------------------------------------------------------------

#Filtro gauss -- imagen con ruido gausiano///////////////////////////////////////////////////////////////////
ft=filter()
t0=time()                                   # toma de tiempo
f_image_gauss=ft.filter_gauss(image_gauss) # filtro gauss a la imagen Gauss
t1=time()                                   # toma de tiempo
cv2.imshow('imagen filtrada con ruido gauss',f_image_gauss)     # visualizar la imagen
image_noiseg_g=np.absolute(image_gauss-f_image_gauss)           # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado gauss con imagen gauss',image_noiseg_g) # visualizar la imagen
print ('raíz de ECM entre imagen lena y lena_gauss filtrada gauss es: ',ft.rmse(image_gray,f_image_gauss))#ECM de la imagen
cv2.waitKey(0)


#Filtro gauss -- imagen con ruido S&P/////////////////////////////////////////////////////////////
t2=time()                                   # toma de tiempo
f_image_syp=ft.filter_gauss(image_syp)                   # filtro gauss a la imagen SyP
t3=time()                                   # toma de tiempo
cv2.imshow('imagen filtrada con ruido syp',f_image_syp)  # visualizar la imagen
image_noiseg_syp=np.absolute(image_syp-f_image_syp)      # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado gauss con imagen S&P',image_noiseg_syp)# visualizar la imagen
cv2.waitKey(0)
print ('raíz de ECM entre imagen lena y lena_S&P filtrada gauss es:',ft.rmse(image_gray,f_image_syp,)) #ECM de la imagen


#-----------------------------------------------------------------------------------------------------------------------

##Filtro mediana -- imagen con ruido gausiano///////////////////////////////////////////////
t4=time()                                   #toma de tiempo
fm_image_gauss=ft.filter_median(image_gauss)                        # Filtrar por mediana a la imagen Gauss
t5=time()                                   #toma de tiempo
cv2.imshow('imagen filtrada med con ruido gauss',fm_image_gauss)    # Visualizar la imagen
image_noisemed_g=np.absolute(image_gauss-fm_image_gauss)            # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado mediana con imagen gauss',image_noisemed_g)# Visualizar Imagen
cv2.waitKey(0)
print ('raíz de ECM entre imagen lena y lena_gauss filtrada mediana es:   ',ft.rmse(image_gray,fm_image_gauss))#ECM de la imagen


#Filtro mediana -- imagen con ruido S&P///////////////////////////////////////////////////
t6=time()                                   # toma de tiempo
fm_image_syp=ft.filter_median(image_syp)                            # filtro mediana a la imagen SyP
t7=time()                                   # toma de tiempo
cv2.imshow('imagen filtrada mediana con ruido s&p',fm_image_syp)    # visualizar la imagen
image_noisemed_syp=np.absolute(image_syp-fm_image_syp) # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado mediana con imagen S&P',image_noisemed_syp) # visualizar la imagen
cv2.waitKey(0)
print ('raíz de ECM entre imagen lena y lena_S&P filtrada mediana es:   ',ft.rmse(image_gray,fm_image_syp))#ECM de la imagen


#------------------------------------------------------------------------------------------------------



##Filtro bilateral -- imagen con ruido gausiano ////////////////////////////////////////
t8=time()                                   #toma de tiempo
fb_image_gauss=ft.filter_bilateral(image_gauss)                         # filtro bilateral a la imagen gauss
t9=time()                                   #toma de tiempo
cv2.imshow('imagen filtrada bilateral con ruido gauss',fb_image_gauss)  # visualizar la imagen
image_noisebil_g=np.absolute(image_gauss-fb_image_gauss)  # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado bilateral con imagen gauss',image_noisebil_g)# Visualizar Imagen
cv2.waitKey(0)
print ('raiz de ECM entre imagen gauss y filtrada bilateral es: ',ft.rmse(image_gray,fb_image_gauss))#ECM de la imagen


##Filtro bilateral -- imagen con ruido SyP ///////////////////////////////////////////////////
t10=time()                                   #toma de tiempo
fb_image_syp=ft.filter_bilateral(image_syp)                         # filtro bilateral a la imagen SyP
t11=time()                                   #toma de tiempo
cv2.imshow('imagen filtrada bilateral con ruido syp',fb_image_syp)  # visualizar la imagen
image_noisebil_syp=np.absolute(image_syp-fb_image_syp)  # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado bilateral con imagen S&P',image_noisebil_syp) # visualizar la imagen
cv2.waitKey(0)
print ('raiz de ECM entre imagen S&P y filtrada bilateral es: ',ft.rmse(image_gray,fb_image_syp))#ECM de la imagen


#-----------------------------------------------------------------------------------------------------------------------


# Filtro nlm -- imagen con ruido gausiano //////////////////////////////////////////////////////////
t12 = time()                                   # toma de tiempo
fn_image_gauss=ft.filter_nlm(image_gauss) # filtro nml a la imagen gauss
t13 = time()                                   # toma de tiempo
cv2.imshow('imagen filtrada nlm con ruido gauss',fn_image_gauss)    # visualizar la imagen
image_noisenlm_g=np.absolute(image_gauss-fn_image_gauss)            # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado nlm con imagen gauss',image_noisenlm_g)  # Visualizar Imagen
cv2.waitKey(0)
print ('raíz de ECM entre imagen lena y lena_gauss_noisy filtrada nlm: ',ft.rmse(image_gray,fn_image_gauss))#ECM de la imagen


# Filtro nml -- imagen con ruido SyP /////////////////////////////////////////////////////////////////
t14 = time()                          #toma de tiempo
fn_image_syp=ft.filter_nlm(image_syp) # filtro nml a la imagen SyP
t15 = time()                          #toma de tiemp

cv2.imshow('imagen filtrada nml con ruido syp',fn_image_syp)        # visualizar la imagen
image_noisenlm_syp=np.absolute(image_syp-fn_image_syp)              # Estimacion del ruido ABS Imagen Ruidosa - Imagen filtrada
cv2.imshow('ruido filtrado nlm con imagen S&P',image_noisenlm_syp)  # visualizar la imagen ruido
cv2.waitKey(0)
print ('raiz de ECM entre imagen lena y lena_S&P filtrada nlm: ',ft.rmse(image_gray,fn_image_syp))#ECM de la imagen


#Impresion de tiempos---------------------------------------------------------------------------------------------------

print ('El tiempo Filtro gauss -- imagen con ruido gausiano es :%f' %(t1-t0))
print ('El tiempo Filtro gauss -- imagen con ruido S&P es :%f' %(t3-t2))
print ('El tiempo Filtro mediana -- imagen con ruido gaussiano es :%f' %(t5-t4))
print ('El tiempo Filtro mediana -- imagen con ruido S&P es :%f' %(t7-t6))
print ('El tiempo Filtro bilateral -- imagen con ruido gaussiano es :%f' %(t9-t8))
print ('El tiempo Filtro bilateral -- imagen con ruido S&P es :%f' %(t11-t10))
print ('El tiempo Filtro nml -- imagen con ruido gaussiano es :%f' %(t13-t12))
print ('El tiempo Filtro nml -- imagen con ruido S&P es :%f' %(t15-t14))

# repositorio en Gthub: https://github.com/ErickB24/Taller_3_Proc_Img.git
