# This function will be helpful to load the TSV file in memory and, from it, 
    # generate random samples to build graphs

def read_lines_tsv(path, encoding='utf-8'):
    lines = []
    with open(path, 'rb') as f:
        for line in f:
            lines.append(line.decode(encoding))
    f.closed
    return lines