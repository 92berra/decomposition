'''
    Generate datasets
'''

# Generate Korean font images
python font2img.py --label-file ../labels/50-common-hangul.txt --font-dir ../fonts/source --output-dir ../images/source/source-50
python font2img.py --label-file ../labels/50-common-hangul.txt --font-dir ../fonts/target --output-dir ../images/target/target-50

# Generate component images (separated from Korean font images)
python separator/separator-1type.py
python separator/separator-2type.py
python separator/separator-3type.py
python separator/separator-4type.py
python separator/separator-5type.py
python separator/separator-6type.py

# Check


# Generate combined images
python combine.py