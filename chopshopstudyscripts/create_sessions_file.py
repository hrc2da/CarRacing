from pilotsessions import sessions, users

with open('h1_human_sessions.txt', 'w+') as outfile:
    for i,user in enumerate(users):
        outfile.write(f'{user},{sessions[i]}\n')
