import argparse
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.toolkits import CodeExecutionToolkit, FileWriteToolkit

def main(training_path, output_dir):
    # Create the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.0},
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the tools
    code_toolkit = CodeExecutionToolkit().get_tools()
    file_write_toolkit = FileWriteToolkit().get_tools()
    
    # Create the agent with the tools
    agent = ChatAgent(
        model=model, 
        tools=code_toolkit+file_write_toolkit
    )

    # Create the prompt with the actual paths
    prompt = (
        f"Solve the ML task described in folder {training_path}. "
        f"Do not modify any files in {training_path}. "
        f"All temp or saved files should be located somewhere under {output_dir}. "
        f"Save the predicted results to {output_dir}, result file name should be \"results\", "
        f"the format and extension should be same as the test data file."
    )

    # Run the agent
    response = agent.step(prompt)
    print(response.msgs[0].content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML task solver")
    parser.add_argument("-t", "--training_path", type=str, help="Path to the training data directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory")
    
    args = parser.parse_args()
    main(args.training_path, args.output_dir)
