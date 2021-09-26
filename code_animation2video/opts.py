import argparse
def parse_opts():
    parser = argparse.ArgumentParser(description='animation2video')
    # =========================  Input Configs ==========================
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/checkpoint_animation2video.pth', help='trained model path')
    parser.add_argument('--image_path', type=str, default=r'./test_data/taile.jpg', help='reference image path')
    parser.add_argument('--dense_flow_path', type=str, default=r'./test_data/taile_Fapp.npy', help='reference approximate dense flow path')
    parser.add_argument('--audio_path', type=str, default=r'./test_data/chuanpu.wav', help='input audio path')
    parser.add_argument('--res_path', type=str, default=r'./result', help='result path')
    # =========================  Base Configs ==========================
    parser.add_argument('--input_channel', type=int, default=3, help='input image channels')
    parser.add_argument('--out_channel', type=int, default=3, help='output image channels')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    #=========================  Network Configs ==========================
    parser.add_argument('--encoder_num_down_blocks', type=int, default=2, help='network setting')
    parser.add_argument('--encoder_block_expansion', type=int, default=64, help='network setting')
    parser.add_argument('--encoder_max_features', type=int, default=512, help='network setting')
    parser.add_argument('--num_bottleneck_blocks', type=int, default=2, help='network setting')
    parser.add_argument('--houglass_num_blocks', type=int, default=5, help='network setting')
    parser.add_argument('--houglass_block_expansion', type=int, default=64, help='network setting')
    parser.add_argument('--houglass_max_features', type=int, default=512, help='network setting')
    args = parser.parse_args()

    return args