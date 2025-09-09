import argparse
import os
import gc

from rvc import Config, load_hubert, get_vc, rvc_infer

def get_rvc_model(model_dir):
    model_path = index_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            model_path = os.path.join(model_dir, file)
        elif file.endswith(".index"):
            index_path = os.path.join(model_dir, file)
    if model_path is None:
        raise FileNotFoundError(f"Tidak ditemukan file .pth di folder {model_dir}")
    return model_path, index_path if index_path else ''

def voice_conversion(
    input_path,
    model_dir,
    output_path,
    pitch_change,
    f0_method,
    index_rate,
    filter_radius,
    rms_mix_rate,
    protect,
    crepe_hop_length
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = Config(device, True)

    rvc_model, rvc_index = get_rvc_model(model_dir)
    hubert = load_hubert(device, config.is_half, "/content/Projects/rvc_models/hubert_base.pt")
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model)

    rvc_infer(
        rvc_index,
        index_rate,
        input_path,
        output_path,
        pitch_change,
        f0_method,
        cpt,
        version,
        net_g,
        filter_radius,
        tgt_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        vc,
        hubert
    )

    del hubert, cpt
    gc.collect()
    print(f"[+] Voice conversion selesai. Output disimpan di: {output_path}")

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path file audio input (.wav)")
    parser.add_argument("-m", "--model-dir", required=True, help="Folder model RVC (.pth dan .index)")
    parser.add_argument("-o", "--output", required=True, help="Path file output hasil voice conversion")
    parser.add_argument("-p", "--pitch", type=int, default=0, help="Perubahan pitch (misalnya -1 atau 1)")
    parser.add_argument("--f0", default="rmvpe", choices=["rmvpe", "mangio-crepe", "crepe", "harvest"], help="Algoritma pitch detection")
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--crepe-hop-length", type=int, default=128)

    args = parser.parse_args()

    voice_conversion(
        input_path=args.input,
        model_dir=args.model_dir,
        output_path=args.output,
        pitch_change=args.pitch,
        f0_method=args.f0,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        crepe_hop_length=args.crepe_hop_length
    )

