import os

def znac2f(f):
    return f.split("_")[1]

def ziskaj_skodu(all_img):
    for f in all_img:
        if 'Fabia' in f and not 'S▌Мkoda' in f:
            return znac2f(f)


def repair(url):
    all_img = os.listdir(os.path.join(url, 'images/'))
    nazov = ziskaj_skodu(all_img)
    for f in all_img:
        if 'S▌Мkoda' in f:
            g = f.replace("S▌Мkoda", "Skoda")
        elif nazov in f:
            g = f.replace(nazov, "Skoda")
        else: continue
        os.rename('images/{}'.format(f), 'images/{}'.format(g))

repair(os.getcwd())