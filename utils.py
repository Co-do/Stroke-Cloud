from lxml import etree
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def format(x):
    tree = etree.parse((x))
    root = tree.getroot()
    d = etree.tostring(root[1])
    d = d.decode(encoding='utf_8')
    data = d.split()
    template = data
    return template

def Rebuild(Vectors, template, size, stroke_thickness):
    svg = []
    for i in Vectors:
        template[3] = str(i[0] * size) + ','
        template[4] = str(i[1] * size)
        template[6] = str(i[2] * size) + ','
        template[7] = str(i[3] * size) + ','
        template[8] = str(i[4] * size) + ','
        template[9] = str(i[5] * size)
        template[16] = 'stroke-width="' + str(stroke_thickness) + '"/>\n  '

        #Variable stroke width option
        # template[16] = 'stroke-width="' + str(i[6]) + '"/>\n  '
        svg.append(bytes(' '.join(template), 'utf-8'))
    return svg

def save(s, dim, filename):
    New = etree.XML(
        '<svg width= "{}" height= "{}" version="1.1" xmlns="http://www.w3.org/2000/svg"></svg>'.format(dim, dim))
    for i in s:
        New.append(etree.fromstring(i))
    tree = etree.ElementTree(New)
    tree.write(filename, pretty_print=True)

def filter(stroke):
    values = []
    strokes = stroke.tolist()
    for i in strokes:
        for j in range(len(i)):
            i[j] = (i[j] + 1) / 2
        if max(i) < 1 and min(i) > 0:
            values.append(i)
    return values

def draw(format_path, size, filename, stroke):
    template = format(format_path)
    stroke = stroke[0,:,:]
    data = filter(stroke)
    svg = Rebuild(data, template, size, size / 128)
    save(svg, size, filename)

def sample(samples, steps, model, noise_scheduler, condition, dim_in):
    stroke = torch.randn(1, samples, dim_in).to(device)
    c = condition[0,:]
    for i, t in enumerate(steps):
        t = torch.full((samples,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                residual = model(stroke, t, c)
                stroke = noise_scheduler.step(residual, t[0], stroke)[0]

    return stroke

def l_sample(timesteps, model, noise_scheduler):
    model.eval()
    latent = torch.randn(1, 1, 256).to(device)
    for i, t in enumerate(timesteps):
        t = torch.full((1,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            residual = model(latent, t)
            latent = noise_scheduler.step(residual, t[0], latent)[0]
            #latent =torch.unsqueeze(latent, 0)
    return latent






