def subsets(numbers):
    if numbers == []:
        return [[]]
    x = subsets(numbers[1:])
    return x + [[numbers[0]] + y for y in x]


def subsets_of_given_size(numbers, n):
    return [x for x in subsets(numbers) if len(x)==n]


configuration_set = {{'-bmp', '-gif', '-os2', '-pnm'}, {'-scale 1/2', '-scale 1/4', '-scale 1/8'},
                     {'-dct int', '-dct fast', '-dct float'},
                     {'-dither fs', '-dither ordered', '-dither none'},
                     {'-nosmooth'}}
mode = ['positionConvert', 'Gaussian', 'Median', 'Bilateral', 'SelfDefine']
all_config = configuration_set + mode
config_option = []

#print(subsets_of_given_size(all_config,2))
#for i in range(0,1024):
#    mode = bin(i)[2:].zfill(10)
#    config_vector = [int(x) for x in str(mode)]

