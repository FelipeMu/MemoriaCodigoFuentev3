import os
from PIL import Image
import re
import math

def create_image_packs(input_dir, output_dir, subfolder_name, pack_name):
    # Crear directorio de salida si no existe
    output_path = os.path.join(output_dir, pack_name)
    os.makedirs(output_path, exist_ok=True)

    # Ruta de la subcarpeta de entrada
    input_path = os.path.join(input_dir, subfolder_name)

    # Listar todas las imágenes en la carpeta
    images = [f for f in os.listdir(input_path) if f.endswith('.png')]

    # Ordenar imágenes por número después de "SANOS_"
    def extract_number(filename):
        pattern = fr"{grupo}_(\d+)_"
        match = re.search(pattern, filename)
        return int(match.group(1)) if match else float('inf')

    images.sort(key=extract_number)

    # Ruta completa de cada imagen
    images = [os.path.join(input_path, img) for img in images]

    # Dividir imágenes en grupos de 6
    num_images = len(images)
    num_packs = math.ceil(num_images / 6)

    for i in range(num_packs):
        # Seleccionar las imágenes para este pack
        start_idx = i * 6
        end_idx = min((i + 1) * 6, num_images)
        pack_images = images[start_idx:end_idx]

        # Crear una grilla de 3 filas x 2 columnas
        num_rows = 3
        num_cols = 2
        cell_width = 0
        cell_height = 0

        # Obtener el tamaño de las imágenes
        loaded_images = [Image.open(img) for img in pack_images]
        cell_width = max(img.width for img in loaded_images)
        cell_height = max(img.height for img in loaded_images)

        # Crear un nuevo lienzo para la imagen final
        grid_width = num_cols * cell_width
        grid_height = num_rows * cell_height
        new_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

        # Pegar las imágenes en la grilla
        for idx, img in enumerate(loaded_images):
            row = idx // num_cols
            col = idx % num_cols
            x_offset = col * cell_width
            y_offset = row * cell_height
            new_image.paste(img, (x_offset, y_offset))

        # Guardar la imagen final
        pack_filename = f"pack_{i+1}.png"
        pack_filepath = os.path.join(output_path, pack_filename)
        new_image.save(pack_filepath)

# Directorios principales
# crear grilla grupo:
grupo = "TEC" # o TEC
base_dir = os.path.join(r"D:\TT\Memoria\CodigoFuente\Graphics_VSC_Norm_NNM2d5OI_FD_LR0d001_E300_DO0d2_CoefCorr", f"GraphicsSignalsVSC_NormMinMax_NNM2d5OI_FD_LR0d001_E300_DO0d2_{grupo}")
output_dir = os.path.join(r"D:\TT\Memoria\CodigoFuente\Graphics_VSC_Norm_NNM2d5OI_FD_LR0d001_E300_DO0d2_CoefCorr", f"PACK6_{grupo}")

# Procesar las carpetas signals_VSCd y signals_VSCi
create_image_packs(base_dir, output_dir, "signals_VSCd", "pack6_VSCd")
create_image_packs(base_dir, output_dir, "signals_VSCi", "pack6_VSCi")
