def model(image):
    return (1, 2)

#  테스트 시연


def test(model):
    total_error = 0
    for image, position in test_dataset:
        estimated = model(image)
        error = distance(estimated, position)
        total_error += error
    return total_error
