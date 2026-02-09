# Text Adventure Game

A simple Python-based text adventure game where you explore rooms, collect items, and try to escape!

## Requirements

- Python 3.10 or higher

## How to Run

```bash
python adventure_game.py
```

## How to Play

### Objective

Navigate through the rooms, collect items, and find your way to victory!

### Basic Commands

**Movement:**

- `north` or `n` - Move north
- `south` or `s` - Move south
- `east` or `e` - Move east
- `west` or `w` - Move west

**Items:**

- `take <item>` - Pick up an item
- `use <item>` - Use an item from your inventory
- `inventory` or `i` - View your inventory

**Other:**

- `look` - Look around the current room
- `help` - Show available commands
- `quit` - Exit the game

## Game Map

```
        [Library]
             |
[Kitchen] - [Entrance] - [Treasure Room]
             |                (locked)
         [Other]
```

## 💡 Tips & Hints

1. **Explore everywhere** - Some rooms contain useful items
2. **Check your inventory** - Use `inventory` to see what you're carrying
3. **Read descriptions carefully** - They often contain hints
4. **Try using items** - Some items can unlock new areas
5. **The treasure room is locked** - You'll need to find a key!

## Walkthrough (Spoilers!)

<details>
<summary>Click to reveal solution</summary>

1. Start in the Entrance Hall
2. Go to the Kitchen (west or left)
3. Pick up the Key
4. Return to the Entrance Hall
5. Go to the Treasure Room (east or right)
6. Use the Key to unlock the door
7. Enter and claim your treasure!

</details>

## Troubleshooting

**"Command not recognized"**

- Make sure you're using the correct command format
- Type `help` to see all available commands

**"Can't go that way"**

- Not all rooms are connected in every direction
- Try a different direction or type `look` to see available exits

**"Item not found"**

- Make sure the item exists in the current room
- Use `look` to see what's available

## Customization

Want to modify the game? Open `adventure_game.py` and you can:

- Add more rooms
- Create new items
- Change room descriptions
- Add puzzles or challenges

## Credits

Created with SimpleCoder AI Agent
