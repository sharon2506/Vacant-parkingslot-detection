def build_smart_reference(frames_gray_list):
    stack = np.stack(frames_gray_list, axis=0).astype(np.float32)

    ref_max = np.max(stack, axis=0).astype(np.uint8)
    ref_min = np.min(stack, axis=0).astype(np.uint8)

    return ref_max, ref_min
