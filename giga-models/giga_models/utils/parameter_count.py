def format_size(x, factor=1000):
    if x > 1e8:
        return '{:.1f}G'.format(x / (factor * factor * factor))
    if x > 1e5:
        return '{:.1f}M'.format(x / (factor * factor))
    if x > 1e2:
        return '{:.1f}K'.format(x / factor)
    return str(x)


def parameter_count(model, factor=1000):
    count = 0
    for parameter in model.parameters():
        count += parameter.numel()
    return count, format_size(count, factor)
