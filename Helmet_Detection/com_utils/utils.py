import base64


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("research/"+fileName, 'wb') as f:
        f.write(imgdata)
        f.close()