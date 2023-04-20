from Graph import Graph, Vertex


def write_graph(g):
    text = []
    text.append("Name, X, Y, Radius, Brightness")
    for star in g:
        text.append(star.get_text_format())
    name = g.get_name()
    parts = str(name).split('/')
    new_name = 'star_data/' + parts[-1][:-4] + '.txt'
    with open(new_name, 'w') as f:
        f.write('\n'.join(text))


def write_assignments(a_list, name1, name2):
    text = []
    text.append("G1 id, G2 id, Confidence")
    for a in a_list:
        text.append(a.get_text_format())
    parts1 = name1.split('/')
    parts2 = name2.split('/')
    new_name1 = parts1[-1][:-4]
    new_name2 = parts2[-1][:-4]
    new_name = str(f'{new_name1}, {new_name2}.txt')
    full_path = 'matches/' + new_name
    with open(full_path, 'w+') as f:
        f.write('\n'.join(text))
