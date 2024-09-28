## Information before using
I changed all the paths to prevent possible information leakage.
In order to run the code on ImageNet, you will need to configure the paths to match your own system (search "path/to")

Note that Python files end with '_t' in '/models' and ' /exps'  means using Dual-Arch

## How to Use

1.Install dependencies in  'requirement.txt'

2.Run:

```bash
python main.py --config=./exps/[MODEL NAME].json
```

where [MODEL NAME] should be chosen from `icarl`, `icarl_t`, `wa`, `wa_t`, `der`,  `der_t`, etc.
