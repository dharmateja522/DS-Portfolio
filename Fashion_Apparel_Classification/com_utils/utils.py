import base64


def decodeImage(imgstring, imageLoc):
    imgdata = base64.b64decode(imgstring)

    with open(imageLoc, 'wb') as f:
        f.write(imgdata)
        f.close()
