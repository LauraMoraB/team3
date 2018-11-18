# Painting Retrival Project

## Code Usage

```bash
usage: main.py [-h] [-me {SIFT,ORB,KAZE,SURF,HOG}] [-ma {BFMatcher,Flann}]
               [-rs] [--test | --validate]

optional arguments:
  -h, --help            show this help message and exit
  --test                test excludes validate
  --validate            validate excludes test

General arguments:
  -me {SIFT,ORB,KAZE,SURF,HOG}, --method {SIFT,ORB,KAZE,SURF,HOG}
  -ma {BFMatcher,Flann}, --matcher {BFMatcher,Flann}
  -rs, --rootsift       Only for sift method
```

Default values are:
- Method: SIFT
    - If method SIFT is selected, ROOTSIFT can be applied
- Matcher: BFMatcher
- Validation values are selected


