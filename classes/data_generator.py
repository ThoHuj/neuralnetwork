from torch import Tensor, rand, mean, tensor


class DataGenerator:
    def generate_random_image_data(self, batch_size: int) -> tuple[Tensor, Tensor]:
        random_image_data_vector = rand(batch_size, 2)
        column_means = mean(random_image_data_vector, dim=1)
        y_true_label_vector = tensor(column_means < 0.5).float().reshape(-1, 1)
        return random_image_data_vector, y_true_label_vector
