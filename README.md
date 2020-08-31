# Jugando con opencv

## OpenCV es una bibioteca de codigo abierto para inteligencia artificial.

* Estas demostraciones se basan en el trabajo genial realizado por GABRIELA SOLANO 
* puedes visitar su blog en
```link
https://omes-va.com/blog/
```
* o su canal de youtube en https://www.youtube.com/channel/UCCDvMED1sysAbF5qOfmEw3A

* Para poder reproducir tu propio algoritmo de emociones debes tener contemplado la carpeta de emojis con los estados de emociones.

* Debes ejecutar los archivos en el siguiente orden
* Nota: recomendable instalar un entorno virtual y ejecutar el archivo requrements.txt ya que esa version de opencv anda bien en mac os x.
1.- captura.py : donde se capturaran cada una de las imagenes de emociones para su posterior calculo, para mac os x realice algunos pequeños ajustes para que funcionara bien como son:

```python
cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)
cap.set(3,640)
cap.set(4,480)
```
* recuerda estos valores son necesarios si tienes 2 webcam o una camara web de alta densidad 
2.- entrenando.py: donde se ejecutara el algoritmo de entrenamiento: el metodo que mayores resultados me dio es LBPH
3.- emociones.py: aqui se detectaran las emociones segun el modelo entrenado 

# Mascarilla
## Para el ejemplo de mascarilla se basa en el archivo haarcascada_mcs_mouth.xml que en conjunto con el default ayuda a detectar correctamente supersociones de objetos.

## Me diverti mucho realizando este proyecto recuerda

### eliminar el archivo rm -rf .DS_Store  para las carpetas que recorras.
### en caso de algun error de opencv ejecutar pip install opencv-contrib-python --upgrade   





