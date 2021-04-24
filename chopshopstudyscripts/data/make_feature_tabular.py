import csv


def process_entry(entry):
    if entry == 'eng_power':
        return 'engine power'
    elif entry == 'friction_lim':
        return 'tire tread'
    elif entry == 'wheel_rad':
        return 'wheel radius'
    elif entry == 'wheel_width':
        return 'wheel width'
    elif entry in ['bumper_width1','bumper_width2','hull2_width1','hull2_width2','hull3_width1','hull3_width2','hull3_width3','hull3_width4','spoiler_width1','spoiler_width2']:
        return 'body shape'
    elif entry == 'steering_scalar':
        return 'steering sensitivity'
    elif entry == 'rear_steering_scalar':
        return 'rear steering'
    elif entry == 'brake_scalar':
        return 'brake sensitivity'
    elif entry == 'max_speed':
        return 'max speed'
    elif entry == 'color':
        return 'color'
    else:
        raise(ValueError(f'Unrecognized design feature {entry}'))
    

def row2tab(i,row,n_attempted):
    body_shape = False
    tab = f'{i+1} &'
    for entry in row:
        formatted_name = process_entry(entry)
        if formatted_name == 'body shape' and body_shape == True:
            continue
        else:
            tab += f' {formatted_name},'
            if formatted_name == 'body shape':
                body_shape = True
    return f'{tab[:-1]} & {n_attempted} \\\\ \n\\hline\n'
    

    

with open('human_modified_features.csv','r') as infile:
    reader = csv.reader(infile)
    with open('num_designs_attempted.csv', 'r') as attemptedfile:
        attemptedreader = csv.reader(attemptedfile)
        attempts = []
        for j,a in enumerate(attemptedreader):
            sess,n = map(int,a)
            assert j == sess
            attempts.append(n)
        with open('human_modified_features.tex','w+') as outfile:
            outfile.write('\\begin{center}\n\\begin{tabular}{|c|l|c|}\n\\hline P & Features Modified & Number Designs Attempted\\\\ \n\hline\n')
            for i,row in enumerate(reader):
                outfile.write(row2tab(i,row,attempts[i]))
            outfile.write('\\end{tabular}\n\\end{center}')