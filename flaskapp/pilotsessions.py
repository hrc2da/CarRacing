


users = [
    'a3476ba7c278432db0315eda9546b7a4', #amit 0 z
    'e6900ed30d77497a97b8b9800d3becdf', #dan 1
    'b15a47a3828c43d79fa74ca0cffdeb53', #alap 2 z
    'b0e35b9e8db847d992fa81afa8851753', #swati 3 z
    '9a5f5d937d79438daa2b52cb4ce26216', #yuhan 4
    '540d7a5f797745c3beeababa9048d930', #jiyhun 5 z
    '2a532da3d761421890cc5de28b3ff2f3', #anna 6  z
    'b00a73908f3147b5b35e90936134a77f', #nikhil 7 z 9

    '786f2281540e4e468fca9d1a74df5c38', #george 8 -- feature confidence z 13 z 1
    'c79f3969753e4d91a9cb88d4382c9ca8', #gonzalo 9 -- feature confidence z 1
    'b11c5f8063d647b1aa73cb00eab176e0', #george 10 friend 1 -- feature confidence z 1 needs to run retrain on human and full_bo_on_redesign
    '1b7db0cc7094434b85c284547269f99c' #george 11 friend 2 -- feature confidence z 1

]

sessions = [
    '5f6d02c2673446edf0d88f1c', #amit
    '5f6cd5e2d8a6d9430d007bf3', #dan
    '5f875228b9d252ffa7e54be9', #alap
    '5f87a26bb9d252ffa7e54bea', #swati
    '5f87bcceb9d252ffa7e54beb', #yuhan
    '5f8df42ad5c68de5e8185f59', #jihyun
    '5f90983b8dfbb43775149641', #anna
    '5f90fdee8dfbb43775149642', #nikhil
    '5fb329dfcf951ee063d04132', #george -- feature confidence
    '5fb33f12cf951ee063d04133', #gonzalo -- feature confidence
    '5fb43427cf951ee063d04135', #george friend 1 -- feature confidence
    '5fb44bafcf951ee063d04136' #george friend 2 -- feature confidence
]

keys = [
    'amit',
    'dan',
    'alap',
    'swati',
    'yuhan',
    'jihyun',
    'anna',
    'nikhil',
    'george',
    'gonzalo',
    'george_f1',
    'george_f2'
]

session2keydict = {sessions[i]:k for i,k in enumerate(keys)}