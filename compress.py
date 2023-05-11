
# Use pandas groupby operations with the average aggregate function
# to compress an image to a new_width by new_width pixel image.

import numpy as np
import pandas as pd

def image_compress(image, new_width):

    spacings = []

    for i in range(2):

        min_val = image.shape[i] // new_width
        rem = image.shape[i] - new_width * min_val

        if rem > 0:

            spacing = new_width // rem
            lim = rem * spacing

            arr = np.arange(new_width) % spacing == 0
            arr = arr + min_val
            arr[lim:] = min_val

        else:
            arr = np.full((new_width,), min_val)

        spacings.append(arr)

    compressed = []

    for s in range(3):

        df = pd.DataFrame(image[:,:,s])
        aggs = []

        for i in range(2):

            agg = []
            counter = 0

            for num in spacings[i]:
                for _ in range(num):
                    agg.append(counter)
                counter += 1

            aggs.append(agg)

        for i in range(2):

            df['agg'] = aggs[i]

            df = df.groupby('agg').mean()
            df = df.reset_index()

            df = df.drop('agg', axis=1)
            df = df.T

        compressed.append(df.to_numpy())

    compressed = np.stack(compressed, axis=-1)
    compressed = compressed.astype(np.uint8)

    return compressed
