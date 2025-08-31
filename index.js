var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/profile.ts
var profile_exports = {};
__export(profile_exports, {
  handler: () => handler
});
module.exports = __toCommonJS(profile_exports);
var import_client_dynamodb = require("@aws-sdk/client-dynamodb");
var import_lib_dynamodb = require("@aws-sdk/lib-dynamodb");
var readline = __toESM(require("readline"));
var client = new import_client_dynamodb.DynamoDBClient({});
var docClient = import_lib_dynamodb.DynamoDBDocumentClient.from(client);
var TABLE_NAME = "newRegister";
async function register(name, username, password) {
  if (!name || !username || !password) {
    throw new Error("Please enter valid values for name, username, and password");
  }
  try {
    const getCommand = new import_lib_dynamodb.GetCommand({
      TableName: TABLE_NAME,
      Key: { username }
    });
    const existingUser = await docClient.send(getCommand);
    if (existingUser.Item) {
      throw new Error("This username already exists, please try a different one!");
    }
    const putCommand = new import_lib_dynamodb.PutCommand({
      TableName: TABLE_NAME,
      Item: {
        name,
        username,
        password,
        createdAt: (/* @__PURE__ */ new Date()).toISOString()
      }
    });
    await docClient.send(putCommand);
    return `Hello! User ${name} (${username}) registered successfully!`;
  } catch (error) {
    throw new Error(error.message);
  }
}
async function login(username, password) {
  try {
    const command = new import_lib_dynamodb.GetCommand({
      TableName: TABLE_NAME,
      Key: { username }
    });
    const response = await docClient.send(command);
    if (response.Item && response.Item.password === password) {
      return response.Item.name;
    }
    throw new Error("Incorrect username or password, please try again!");
  } catch (error) {
    throw new Error(error.message);
  }
}
function getUserInput(question) {
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
    if (action === "exit") {
      console.log("Exiting");
      break;
    }
    if (action === "register") {
      const name = await getUserInput("Enter name: ");
      const username = await getUserInput("Enter username: ");
      const password = await getUserInput("Enter password: ");
      await register(name, username, password);
    } else if (action === "login") {
      const username = await getUserInput("Enter username: ");
      const password = await getUserInput("Enter password: ");
      await login(username, password);
    } else {
      console.log("Invalid action. Please try again.");
    }
  }
}
var handler = async (event, context) => {
  const { action, name, username, password } = event;
  try {
    switch (action) {
      case "register":
        if (!name || !username || !password) {
          throw new Error("Register requires name, username, and password");
        }
        return await register(name, username, password);
      case "login":
        if (!username || !password) {
          throw new Error("Login requires username and password");
        }
        const userName = await login(username, password);
        return `Successfully logged in, welcome! (${userName})`;
      default:
        throw new Error("Invalid action. Use 'register' or 'login'");
    }
  } catch (error) {
    return {
      statusCode: 400,
      body: JSON.stringify({ error: error.message })
    };
  }
};
if (require.main === module) {
  main().catch((error) => console.error("Error in main:", error));
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  handler
});
