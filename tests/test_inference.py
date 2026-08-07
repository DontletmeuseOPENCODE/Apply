def test_dataset_generation():
    import json
    import os
    train_path = "data/processed/train.jsonl"
    if not os.path.exists(train_path):
        return
    with open(train_path) as f:
        lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        assert "messages" in item
        assert len(item["messages"]) >= 2
    assert len(lines) > 0
