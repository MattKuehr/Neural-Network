import Model_v2 as model
import numpy as np

# overriding numpy's dot() function for consistency across machines
# DO NOT DELETE THIS
model.np_init()

# creating directory to save model parameters and objects
save_path = model.create_directory()


manual_setup = True

if manual_setup:
    # Importing dataset
    path = r"C:\Users\Matt\Desktop\Research\example_file"
    reader = model.CustomCSVReader(filepath=path, delimiter=',')

    # Get input features
    X = reader.get_columns(column_names=['var1', 'var2', 'var3'])

    # Get output feature
    y = reader.get_column('class').astype(int) -1

    # Split between training data and validation / testing data
    X_train, X_temp, y_train, y_temp = model.custom_train_test_split(X, y, test_size=0.3, random_state=42)

    # Split validation / testing data
    X_val, X_test, y_val, y_test = model.custom_train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalizing data
    X_train, X_test, X_val = model.normalize(X_train, X_test, X_val, has_val=True)
    
    # Building classifier
    #nn = model.Model()
    #nn.add(model.Layer_Dense(3, 32))
    #nn.add(model.Activation_ReLU())
    #nn.add(model.Layer_Dense(32, 18))
    #nn.add(model.Activation_Softmax())
    #nn.set(
        #loss=model.Loss_CategoricalCrossentropy(),
        #optimizer=model.Optimizer_Adam(learning_rate=0.001, decay=0,
                                       #epsilon=1e-7, beta_1=0.9,
                                       #beta_2=0.999),
        #accuracy=model.Accuracy_Categorical()
    #)
    #nn.finalize()

    # Train the model
    #nn.train(X_train, y_train, validation_data=(X_val, y_val),
            #epochs=1000, batch_size=128, print_every=100)
    
    # Load model params instead of training
    #nn.load_parameters('Test Params')

    # Load entire model instead of building and training
    nn = model.Model.load('Test Model')
    
    # Evaluate on test data
    nn.evaluate(X_test, y_test)

    # Make predictions on testing data (this could be used for plotting)
    #nn.predict(X_test, batch_size=128)

    # Method 1 to save the model: Save weight and biases
    #nn.save_parameters('Test Params')

    # Method 2 to save the model: Save entire model object
    #nn.save('Test Model')

else:
    if __name__ == '__main__':
        
        # collecting CSV file path 
        path = input("Enter the path of the file: ")
        path = path[1:len(path)-1] 
        path = rf"{path}"
        reader = model.CustomCSVReader(filepath=path, delimiter=',')
        
        # collecting names of input columns
        inputs = []
        inputting = True
        while inputting:
            input_ = input("Enter name of the input column (Case sensitive): \nEnter 'done' to stop: ").strip()
            if input_.lower() == 'done':
                inputting = False
            else:
                inputs.append(input_)
        
        if len(inputs) == 0:
            print("No input columns provided.")
            exit(0)
        elif len(inputs) == 1:
            X = reader.get_column(inputs[0])
        else:
            X = reader.get_columns(inputs)

        # collecting name of output column
        output_ = input("Enter name of the output column (Case sensitive): ").strip()
        
        # model choice
        model_choice = int(input("Enter 1 for classification, 2 for regression: ").strip())

        # get number of classes for classification
        if model_choice == 1:
            num_classes = int(input("Enter the number of classes: ").strip())
            is_classifier = True
        elif model_choice == 2:
            is_classifier = False
        else:
            print("Invalid choice")
            exit(0)

        if is_classifier:
            y = reader.get_column(output_).astype(int) - 1
        else:
            y = reader.get_column(output_, flatten=False)

        # splitting data into training and testing sets
        val = input('Do you want validation data? (y/n): ').lower().strip()
        has_val = False
        if val == 'y':
            train_percent = int(input("What percentage of data do you want for training? The rest will be split among testing / validation. ").strip())
            val_percent = int(input("What percentage of the remaining data do you want for validation? The rest will be used for testing. ").strip())
            has_val = True
        else:
            print("Enter your data split percentages in the format: train, test")
            split = input('Enter the data split percentages, e.g. 70, 30: ').strip().split(',')
        
        if has_val:
            X_train, X_temp, y_train, y_temp = model.custom_train_test_split(X=X, y=y, test_size=1.0 - (train_percent/100.0), random_state=42)
            X_val, X_test, y_val, y_test = model.custom_train_test_split(X_temp, y_temp, test_size=1.0 - (val_percent/100.0), random_state=42) 
        else:
            X_train, X_test, y_train, y_test = model.custom_train_test_split(X=X, y=y, test_size=1.0-(train_percent/100.0), random_state=42)

        # normalize data ?
        normalization = input("Do you want to normalize the data? (y/n): ").lower().strip()
        if normalization == 'y':
            
            X_train, X_test, X_val = model.normalize(X_train=X_train, X_test=X_test, 
                                                     X_val=X_val, has_val=True)

            '''
            # Extracting max, min and range from training data 
            # to use for normalization
            X_train_max = np.max(X_train, axis=0)
            X_train_min = np.min(X_train, axis=0)
            train_range = X_train_max - X_train_min

            # Normalizing training, validation and testing data
            X_train = 2 * ((X_train - X_train_min) / train_range) -1 
            X_test = 2 * ((X_test - X_train_min) / train_range) -1

            # Normalizing validation data if it exists
            if has_val:
                X_val = 2 * ((X_val - X_train_min) / train_range) -1
            '''
                
        # create model
        nn = model.Model()

        # get number of layers
        num_layers = int(input("Enter the number of layers: ").strip())

        # get number of neurons in each layer
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append((len(inputs), int(input('Enter the number of neurons in the first layer: ').strip())))
            elif i>0 and i < num_layers-1:
                layers.append((layers[i-1][1], int(input(f'Enter the number of neurons in layer {i+1}: ').strip())))
            else:
                if is_classifier:
                    layers.append((layers[i-1][1], num_classes))
                else:
                    layers.append((layers[i-1][1], 1))
        
        # choose optimizer
        optim_choice = int(input("Enter 1 for SGD, 2 for Adam: ").strip())
        print('To edit hyperparameters, manually enter them into the object declarations on line 111 / 113.')

        if optim_choice == 1:
            optim = model.Optimizer_SGD(learning_rate=1, decay=0, momentum=0)
        elif optim_choice == 2:
            optim = model.Optimizer_Adam(learning_rate=0.05, decay=1e-4, epsilon=1e-7, beta_1=0.9, beta_2=0.999)
        else:
            print("Invalid choice")
            exit(0)
        
        # build model
        if is_classifier:
            for i in range(num_layers - 1):
                nn.add(model.Layer_Dense(n_inputs=layers[i][0], n_neurons=layers[i][1]))
                nn.add(model.Activation_ReLU())
            nn.add(model.Layer_Dense(layers[-1][0], layers[-1][1]))
            nn.add(model.Activation_Softmax())
            nn.set(loss=model.Loss_CategoricalCrossentropy(),
                   optimizer=optim,
                   accuracy=model.Accuracy_Categorical()
            )
        else:
            for i in range(num_layers - 1):
                nn.add(model.Layer_Dense(n_inputs=layers[i][0], n_neurons=layers[i][1]))
                nn.add(model.Activation_ReLU())
            nn.add(model.Layer_Dense(layers[-1][0], layers[-1][1]))
            nn.add(model.Activation_Linear())
            nn.set(loss=model.Loss_MeanSquaredError(),
                   optimizer=optim,
                   accuracy=model.Accuracy_Regression()
            )
        
        nn.finalize()

        # get number of epochs and batch size
        epoch_choice = int(input("Enter the number of epochs: ").strip())
        batch_size_choice = int(input("Enter the batch size (default is 128): ").strip())
        
        # train model
        if val:
            nn.train(X_train, y_train, validation_data=(X_val, y_val),
                epochs=epoch_choice, batch_size=batch_size_choice, print_every=100)
        else:
            nn.train(X_train, y_train, epochs=epoch_choice, 
                     batch_size=batch_size_choice, print_every=100)

        
        # Evaluate on test data
        nn.evaluate(X_test, y_test)

        # Now predict on test data
        nn.predict(X_test, batch_size=batch_size_choice)

        # ask the user if they want to save the model
        model.save_prompt(model=nn, path=save_path)

