import torch
from torchvision import datasets, transforms
from models.gaussian_process_layer import RandomFeatureGaussianProcess
from models.trainer import Trainer

if __name__ == "__main__":
    # Step 1: Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        # flatten
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Step 2: Define the SNGP model configuration
    sngp_config = {
        'out_features': 10,  # Number of classes for MNIST
        'backbone': None,  # Use the default ResNetBackbone
        'num_inducing': 1024,
        'momentum': 0.9,
        'ridge_penalty': 1e-6,
    }

    # Step 3: Define training configuration
    training_config = {
        'batch_size': 64,
        'shuffle': True
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(
        model_config=sngp_config,
        task_type='classification',
        model=RandomFeatureGaussianProcess,
        device=device
    )

    # Step 4: Choose whether to train or load the model
    train_model = False  # Set to False to load a pre-trained model
    model_path = './sngp_mnist_model.pth'

    if train_model:
        # Train the model
        for _ in trainer.train(
                training_data=train_dataset,
                data_loader_config=training_config,
                epochs=10,
                lr=0.001
        ):
            pass  # Fully consume the generator to complete training

        # Save the trained model
        torch.save(trainer.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Step 5: Evaluate the model
        trainer.plot_loss("MNIST Classification")
    else:
        # Load the pre-trained model
        trainer.model.load_state_dict(torch.load(model_path, map_location=device))
        trainer.model.eval()
        print(f"Model loaded from {model_path}")

