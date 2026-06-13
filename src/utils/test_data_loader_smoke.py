from src.utils.data_loader import RadarDataLoader


def main():
    loader = RadarDataLoader()

    print("=" * 80)
    print("RADAR LOADER VALIDATION")
    print("=" * 80)
    print(loader.validate())

    print("\nFirst 3 sample IDs:")
    sample_ids = loader.list_sample_ids(limit=3)
    print(sample_ids)

    print("\nLoading first 3 samples:")
    for sample_id in sample_ids:
        trace = loader.load_sample(sample_id)
        print("-" * 80)
        print("sample_id:", trace.sample_id)
        print("binary_label:", trace.binary_label)
        print("category:", trace.category)
        print("family:", trace.family)
        print("full_label:", trace.full_label)
        print("timesteps:", trace.num_timesteps)
        print("event_shape:", trace.events.shape)
        print("first_5_events:", list(trace.events.columns[:5]))


if __name__ == "__main__":
    main()
