# NanoChat UI JavaScript Functions Summary

## 功能模块关系图

```mermaid
graph TD
    A[UI初始化] --> B[事件监听器设置]
    A --> C[全局变量初始化]
    
    B --> D[输入框事件监听]
    B --> E[键盘事件监听]
    B --> F[发送按钮事件监听]
    B --> G[新对话按钮事件监听]
    
    C --> H[消息历史数组]
    C --> I[生成状态标志]
    C --> J[温度参数]
    C --> K[Top-K参数]
    
    D --> DA[自动调整输入框高度]
    D --> DB[控制发送按钮状态]
    
    E --> EA[Enter发送消息]
    E --> EB[Ctrl+Shift+N新建对话]
    
    F --> FA[sendMessage函数]
    
    G --> GA[newConversation函数]
    
    FA --> FB[handleSlashCommand]
    FA --> FC[generateAssistantResponse]
    FA --> FD[addMessage]
    
    FB --> FBA[/temperature命令]
    FB --> FBB[/topk命令]
    FB --> FBC[/clear命令]
    FB --> FBD[/help命令]
    
    FC --> FCA[API请求构建]
    FC --> FCB[流式响应处理]
    FC --> FCC[实时显示更新]
    FC --> FCD[错误处理]
    
    FD --> FDA[用户消息显示]
    FD --> FDB[助手消息显示]
    FD --> FDC[控制台消息显示]
    
    GA --> GAA[清空消息历史]
    GA --> GAB[重置界面状态]
    GA --> GAC[聚焦输入框]
    
    FC --> FE[editMessage]
    FC --> FF[regenerateMessage]
    
    FE --> FEA[编辑历史消息]
    FE --> FEB[重新开始对话]
    
    FF --> FFA[重新生成回复]
    FF --> FFB[清理后续消息]
```

## 详细功能说明

### 核心功能模块

#### 1. 初始化模块
- **全局变量初始化**：设置消息历史、生成状态、温度参数、Top-K参数等
- **DOM元素引用**：获取聊天容器、输入框、发送按钮等元素引用
- **初始状态设置**：设置发送按钮初始状态、输入框自动聚焦

#### 2. 事件监听模块
- **输入事件监听**：监听输入框内容变化，自动调整高度并控制发送按钮状态
- **键盘事件监听**：处理Enter发送消息和Ctrl+Shift+N新建对话快捷键
- **点击事件监听**：处理发送按钮和新对话按钮点击事件

#### 3. 对话管理模块
- **newConversation()**：创建新对话，清空历史记录和界面
- **addMessage()**：向界面添加消息，支持用户、助手和控制台三类消息
- **editMessage()**：编辑历史消息，支持从指定点重新开始对话
- **regenerateMessage()**：重新生成助手回复

#### 4. 消息发送模块
- **sendMessage()**：主发送函数，处理用户输入并触发回复生成
- **handleSlashCommand()**：处理斜杠命令（/temperature、/topk、/clear、/help）

#### 5. 助手回复生成模块
- **generateAssistantResponse()**：生成助手回复，处理API请求和流式响应
  - 构建API请求参数
  - 处理流式响应数据
  - 实时更新界面显示
  - 处理错误情况

### 数据流说明

1. **用户输入流程**：
   用户在输入框输入内容 → 按Enter或点击发送按钮 → sendMessage()处理 → 添加到消息历史 → 调用generateAssistantResponse()生成回复

2. **助手回复流程**：
   generateAssistantResponse()发起API请求 → 接收流式响应 → 逐段解析并显示 → 完成后更新消息历史

3. **命令处理流程**：
   用户输入以/开头的命令 → handleSlashCommand()解析 → 执行对应命令逻辑 → 显示结果

4. **对话管理流程**：
   点击新对话按钮 → newConversation()清空状态
   点击用户消息 → editMessage()重新编辑
   点击助手消息 → regenerateMessage()重新生成

### 状态管理

- **isGenerating**：标识是否正在生成回复，影响界面交互状态
- **messages**：存储完整的对话历史，用于API请求
- **currentTemperature**：控制生成随机性参数
- **currentTopK**：控制词汇选择范围参数

### UI交互特性

- **实时高度调整**：输入框根据内容自动调整高度
- **智能按钮状态**：根据输入内容和生成状态控制发送按钮
- **流式响应显示**：逐步显示助手回复，提供更好的用户体验
- **消息编辑功能**：支持重新编辑历史消息并从该点继续对话
- **回复重新生成**：支持重新生成助手回复
- **键盘快捷操作**：支持快捷键操作提升使用效率