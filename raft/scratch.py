total_steps = 0
VAL_FREQ = 5000
while total_steps < 10000:
    if total_steps % VAL_FREQ == VAL_FREQ - 1:
        print(total_steps)
    total_steps += 1
