import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, GetCommand } from "@aws-sdk/lib-dynamodb";
import * as readline from 'readline';

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);
const TABLE_NAME = "newRegister";

async function register(name: string, username: string, password: string): Promise<string> {
  if (!name || !username || !password) {
    throw new Error("Please enter valid values for name, username, and password");
  }
  
  try {
    // Check if username already exists
    const getCommand = new GetCommand({
      TableName: TABLE_NAME,
      Key: { username: username },
    });

    const existingUser = await docClient.send(getCommand);
    
    if (existingUser.Item) {
      throw new Error("This username already exists, please try a different one!");
    }

    // Insert new user
    const putCommand = new PutCommand({
      TableName: TABLE_NAME,
      Item: {
        name,
        username,
        password,
        createdAt: new Date().toISOString(),
      },
    });

    await docClient.send(putCommand);
    return `Hello! User ${name} (${username}) registered successfully!`;
  } catch (error: any) {
    throw new Error(error.message);
  }
}

async function login(username: string, password: string): Promise<string> {
  try {
    const command = new GetCommand({
      TableName: TABLE_NAME,
      Key: { username: username },
    });

    const response = await docClient.send(command);
    
    if (response.Item && response.Item.password === password) {
      return response.Item.name;
    }

    throw new Error("Incorrect username or password, please try again!");
  } catch (error: any) {
    throw new Error(error.message);
  }
}

function getUserInput(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

async function main() {
  while (true) {
    const action = await getUserInput("Enter action (register/login/exit): ");
    
    if (action === 'exit') {
      console.log("Exiting");
      break;
    }

    if (action === 'register') {
      const name = await getUserInput("Enter name: ");
      const username = await getUserInput("Enter username: ");
      const password = await getUserInput("Enter password: ");
      await register(name, username, password);
    } else if (action === 'login') {
      const username = await getUserInput("Enter username: ");
      const password = await getUserInput("Enter password: ");
      await login(username, password);
    } else {
      console.log("Invalid action. Please try again.");
    }
  }
}

// lambda handler
export const handler = async (event: any, context: any): Promise<any> => {
  const { action, name, username, password } = event;

  try {
    switch (action) {
      case 'register':
        if (!name || !username || !password) {
          throw new Error("Register requires name, username, and password");
        }
        return await register(name, username, password);
      case 'login':
        if (!username || !password) {
          throw new Error("Login requires username and password");
        }
        const userName = await login(username, password);
        return `Successfully logged in, welcome! (${userName})`;
      default:
        throw new Error("Invalid action. Use 'register' or 'login'");
    }
  } catch (error: any) {
    return {
      statusCode: 400,
      body: JSON.stringify({ error: error.message })
    };
  }
};

// not for lambda
if (require.main === module) {
  main().catch(error => console.error("Error in main:", error));
}






