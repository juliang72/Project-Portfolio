import java.util.ArrayList;
import tester.*;
import javalib.impworld.*;
import java.awt.Color;
import javalib.worldimages.*;
import java.util.Arrays;
import java.util.Random;

// represents a card
class Card {
  int rank;
  String suit;
  boolean flip;
  Posn posn;
  boolean found;
  boolean red;

  // constructor
  Card(int rank, String suit, Posn posn) {
    this.rank = rank;
    this.suit = suit;
    this.flip = false;
    this.posn = posn;
    this.found = false;
    if (this.suit.equals("♥") || this.suit.equals("♦")) {
      this.red = true;
    }
    else {
      this.red = false;
    }
  }
  /*
   * Template: Fields ...rank... -- int ...suit... -- String ...flip... -- boolean
   * ...posn... -- posn ...found... -- boolean ...red... -- boolean Methods
   * ...draw()... -- WorldImage ...drawAt(int, int, WorldScene)... -- void
   * ...cardsEqual(Card)... -- boolean ...withinPosition(Posn)... -- boolean
   * ...updatePosition(int, int)... -- void Methods for Fields ...drawAt(int, int,
   * WorldScene)... -- void ...cardsEqual(Card)... -- boolean
   * ...withinPosition(Posn)... -- boolean ...updatePosition(int, int)... -- void
   */

  // draws the individual card
  WorldImage draw() {
    if (this.found) {
      return new TextImage("X", 15, Color.blue);
    }
    else if (!this.flip) {
      return new TextImage("?", 15, Color.black);
    }
    else {
      return new TextImage(Integer.toString(rank) + suit, 15, Color.black);
    }
  }

  // places the drawn card in a specific location on the background
  void drawAt(int col, int row, WorldScene background) {
    background.placeImageXY(this.draw(), col, row);
  }

  // does this card have the same rank as that card (means of measuring two cards'
  // equality)
  boolean cardsEqual(Card that) {
    return this.rank == that.rank && (this.red == that.red);
  }

  // is this card's position close to a position that is passed in
  boolean withinPosition(Posn posn) {
    return posn.x > this.posn.x - 20 && posn.y < this.posn.y + 20 && posn.x < this.posn.x + 20
        && posn.y > this.posn.y - 20;
  }

  // updates the position of the card, which is needed because we shuffle the deck
  // in createBoard
  void updatePosition(int row, int column) {
    this.posn = new Posn(row * 120 + 30, column * 35 + 20);
  }

}

// represents the Concentration card game
class ConcentrationGame {

  ArrayList<ArrayList<Card>> board = new ArrayList<ArrayList<Card>>();
  ArrayList<Card> carddeck = new ArrayList<Card>();

  // constructor 1
  ConcentrationGame() {
    ArrayList<String> suits = new ArrayList<String>(Arrays.asList("♥", "♠", "♦", "♣"));
    for (int i = 0; i < 4; i++) {
      for (int i2 = 0; i2 < 13; i2++) {
        Card card = new Card(i2, suits.get(i), new Posn(i * 120 + 30, i2 * 35 + 20));
        this.carddeck.add(card);
      }
    }
    this.createBoard(new Random());
  }

  // constructor 2 for purpose of testing
  ConcentrationGame(ArrayList<ArrayList<Card>> deck) {
    this.board = deck;
    ArrayList<String> suits = new ArrayList<String>(Arrays.asList("♥", "♠", "♦", "♣"));
    for (int i = 0; i < 4; i++) {
      for (int i2 = 0; i2 < 13; i2++) {
        Card card = new Card(i2, suits.get(i), new Posn(i * 120 + 30, i2 * 35 + 20));
        this.carddeck.add(card);
      }
    }
  }

  /*
   * Template Fields ...board... -- ArrayList<ArrayList<Card>>() ...carddeck... --
   * ArrayList<Card>() Methods ...createBoard(Random)... -- void
   * ...makeScene(WorldScene)... -- WorldScene No methods for fields
   */

  // creates the board with 13 rows and 4 columns, without repeated cards
  void createBoard(Random rand) {
    board.clear();
    for (int row = 0; row < 4; row++) {
      ArrayList<Card> rowlist = new ArrayList<Card>();
      for (int column = 0; column < 13; column++) {
        Card givenCard = carddeck.get(rand.nextInt(52));
        if (new ArrayUtils().containsCard(board, givenCard) || rowlist.contains(givenCard)) {
          column--;
        }
        else {
          rowlist.add(givenCard);
          givenCard.updatePosition(row, column);
        }

      }
      board.add(rowlist);
    }
  }

  // creates the scene with the board of cards
  WorldScene makeScene(WorldScene initialScene) {
    for (int row = 0; row < board.size(); row++) {
      for (int column = 0; column < board.get(0).size(); column++) {
        Card placedCard = board.get(row).get(column);
        placedCard.drawAt(row * 120 + 30, column * 35 + 20, initialScene);
      }
    }
    return initialScene;
  }

}

// holds custom ArrayList methods
class ArrayUtils {

  /*
   * Template No fields Methods ...containsCard(ArrayList<ArrayList<Card>>,
   * Card)... -- boolean No methods for fields
   */

  // does this 2d arraylist of cards contain this card
  boolean containsCard(ArrayList<ArrayList<Card>> arr, Card card) {
    for (ArrayList<Card> row : arr) {
      if (row.contains(card)) {
        return true;
      }
    }
    return false;
  }
}

// renders the game
class GameWorld extends World {
  ConcentrationGame game;
  int pairsleft;
  int timer;
  int remaining;

  // constructor
  GameWorld() {
    this.game = new ConcentrationGame();
    this.pairsleft = 26;
    this.timer = 0;
    this.remaining = 1000;

  }

  /*
   * Template Fields ...game... -- ConcentrationGame ...pairsleft... -- int
   * ...timer... -- int Methods ...makeScene()... -- WorldScene ...worldEnds()...
   * -- WorldEnd ... ...makeEndScene()... -- WorldScene ...OnMouseClicked(Posn)...
   * -- void ... ...onTick()... -- void Methods for Fields ...makeScene()... --
   * WorldScene ...worldEnds()... -- WorldEnd ... ...OnMouseClicked(Posn)... --
   * void ... ...onTick()... -- void
   * 
   */

  // creates the scene of the Concentration game
  public WorldScene makeScene() {
    WorldScene initialScene = new WorldScene(500, 500);
    initialScene.placeImageXY(
        new TextImage("Pairs Remaining: " + Integer.toString(this.pairsleft), 18, Color.black), 100,
        480);
    initialScene.placeImageXY(
        new TextImage("Timer: " + Integer.toString(this.timer / 2), 18, Color.green), 240, 480);
    initialScene.placeImageXY(
        new TextImage("Remaining: " + Integer.toString(this.remaining), 18, Color.blue), 380, 480);
    return this.game.makeScene(initialScene);

  }

  // determines if the game is ended
  public WorldEnd worldEnds() {
    ArrayList<Card> foundcards = new ArrayList<Card>();
    for (int row = 0; row < this.game.board.size(); row++) {
      for (int column = 0; column < this.game.board.get(0).size(); column++) {
        if (this.game.board.get(row).get(column).found) {
          foundcards.add(this.game.board.get(row).get(column));
        }
      }

    }
    if (foundcards.size() == 52 || this.remaining == 0) {
      return new WorldEnd(true, this.makeEndScene());
    }
    else {
      return new WorldEnd(false, this.makeScene());
    }
  }

  // creates the last scene when the game is completed
  public WorldScene makeEndScene() {
    if (this.remaining > 0) {
      WorldScene initialScene = new WorldScene(500, 500);
      initialScene.placeImageXY(new TextImage("You've Won", 60, Color.green), 250, 250);
      initialScene.placeImageXY(
          new TextImage("Time: " + Integer.toString(this.timer / 2), 40, Color.green), 250, 400);
      return initialScene;
    }
    else {
      WorldScene initialScene = new WorldScene(500, 500);
      initialScene.placeImageXY(new TextImage("You've Lost", 60, Color.red), 250, 250);
      initialScene.placeImageXY(
          new TextImage("Time: " + Integer.toString(this.timer / 2), 40, Color.red), 250, 400);
      return initialScene;
    }

  }

  // flips the cards that are clicked on
  public void onMouseClicked(Posn posn) {
    for (int row = 0; row < this.game.board.size(); row++) {
      for (int column = 0; column < this.game.board.get(0).size(); column++) {
        if (this.game.board.get(row).get(column).withinPosition(posn)
            && !this.game.board.get(row).get(column).found) {
          this.game.board.get(row).get(column).flip = !this.game.board.get(row).get(column).flip;
          this.remaining--;
        }
      }
    }
  }

  // updates the GameWorld, and provides game functionality for flipped cards
  public void onTick() {
    ArrayList<Card> flippedcards = new ArrayList<Card>();
    for (int row = 0; row < this.game.board.size(); row++) {
      for (int column = 0; column < this.game.board.get(0).size(); column++) {
        if (this.game.board.get(row).get(column).flip) {
          flippedcards.add(this.game.board.get(row).get(column));
        }
      }
    }
    if (flippedcards.size() == 2) {
      if (flippedcards.get(0).cardsEqual(flippedcards.get(1))) {
        flippedcards.get(0).found = true;
        flippedcards.get(1).found = true;
        flippedcards.get(0).flip = false;
        flippedcards.get(1).flip = false;
      }
      else {
        flippedcards.get(0).flip = false;
        flippedcards.get(1).flip = false;

      }

    }
    ArrayList<Card> foundcards = new ArrayList<Card>();
    for (int row = 0; row < game.board.size(); row++) {
      for (int column = 0; column < game.board.get(0).size(); column++) {
        if (game.board.get(row).get(column).found) {
          foundcards.add(game.board.get(row).get(column));
        }
      }

    }
    this.pairsleft = (this.game.carddeck.size() - foundcards.size()) / 2;
    this.timer++;
  }

}

// examples and tests
class ExamplesCards {
  void testGame(Tester t) {
    GameWorld g = new GameWorld();
    g.makeScene();
  }

  // tests big bang
  void testBigBang(Tester t) {
    GameWorld g = new GameWorld();
    int worldWidth = 500;
    int worldHeight = 500;
    double tickRate = 0.5;
    g.bigBang(worldWidth, worldHeight, tickRate);
  }

  // tests the makeScene method
  void testmakeScene(Tester t) {
    WorldScene initialScene = new WorldScene(500, 500);
    ArrayList<ArrayList<Card>> exampleDeck = new ArrayList<ArrayList<Card>>(
        Arrays.asList(new ArrayList<Card>(Arrays.asList((new Card(1, "♥", new Posn(0, 0))),
            (new Card(2, "♠", new Posn(0, 0)))))));
    ConcentrationGame game1 = new ConcentrationGame(exampleDeck);
    WorldScene expectedScene = new WorldScene(500, 500);
    expectedScene.placeImageXY(new Card(1, "♥", new Posn(0, 0)).draw(), 30, 20);
    expectedScene.placeImageXY(new Card(2, "♠", new Posn(0, 0)).draw(), 30, 55);
    t.checkExpect(game1.makeScene(initialScene), expectedScene);

  }

  // tests the draw method (Add in other cases)
  void testDraw(Tester t) {
    Card card1 = new Card(1, "♥", new Posn(0, 0));
    card1.flip = true;
    Card card2 = new Card(12, "♠", new Posn(0, 0));
    card2.flip = true;
    t.checkExpect(card1.draw(), new TextImage(Integer.toString(1) + "♥", 15, Color.black));
    t.checkExpect(card2.draw(), new TextImage(Integer.toString(12) + "♠", 15, Color.black));
    card1.flip = false;
    card2.flip = false;
    t.checkExpect(card1.draw(), new TextImage("?", 15, Color.black));
    t.checkExpect(card2.draw(), new TextImage("?", 15, Color.black));
    card1.found = true;
    card2.found = true;
    t.checkExpect(card1.draw(), new TextImage("X", 15, Color.blue));
    t.checkExpect(card2.draw(), new TextImage("X", 15, Color.blue));
  }

  // tests the drawAt method
  void testDrawAt(Tester t) {
    Card card1 = new Card(1, "♥", new Posn(0, 0));
    WorldScene initialScene = new WorldScene(500, 500);
    WorldScene expectedScene = new WorldScene(500, 500);
    expectedScene.placeImageXY(card1.draw(), 0, 0);
    card1.drawAt(0, 0, initialScene);
    t.checkExpect(initialScene, expectedScene);
  }

  // tests the createBoard method
  void testcreateBoard(Tester t) {
    ArrayList<ArrayList<Card>> exampleDeck = new ArrayList<ArrayList<Card>>(
        Arrays.asList(new ArrayList<Card>(Arrays.asList((new Card(1, "♥", new Posn(0, 0))),
            (new Card(2, "♠", new Posn(0, 0)))))));
    ConcentrationGame game1 = new ConcentrationGame(exampleDeck);
    // game1.createBoard(new Random(1));
    t.checkExpect(game1.board.size(), 1);
    t.checkExpect(game1.board, Arrays.asList(new ArrayList<Card>(
        Arrays.asList((new Card(1, "♥", new Posn(0, 0))), (new Card(2, "♠", new Posn(0, 0)))))));
  }

  // tests the containsCard method
  void testContainsCard(Tester t) {
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(0, 0));
    t.checkExpect(new ArrayUtils().containsCard(new ArrayList<ArrayList<Card>>(
        Arrays.asList(new ArrayList<Card>(Arrays.asList(card, card2)))), card), true);
    t.checkExpect(new ArrayUtils().containsCard(
        new ArrayList<ArrayList<Card>>(Arrays.asList(new ArrayList<Card>(Arrays.asList((card2))))),
        card), false);
  }

  // tests the cardsEqual method
  void testCardsEqual(Tester t) {
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(0, 0));
    Card card3 = new Card(2, "♥", new Posn(0, 0));
    Card card4 = new Card(2, "♦", new Posn(0, 0));
    Card card5 = new Card(2, "♣", new Posn(0, 0));
    t.checkExpect(card.cardsEqual(card2), false);
    t.checkExpect(card.cardsEqual(card3), true);
    t.checkExpect(card.cardsEqual(card4), true);
    t.checkExpect(card.cardsEqual(card5), false);
  }

  // tests the withinPosition method
  void testwithinPosition(Tester t) {
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(500, 500));
    t.checkExpect(card.withinPosition(new Posn(500, 500)), false);
    t.checkExpect(card2.withinPosition(new Posn(500, 500)), true);
    t.checkExpect(card2.withinPosition(new Posn(490, 490)), true);
  }

  // tests the updatePosition method
  void testupdatePosition(Tester t) {
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(500, 500));
    card.updatePosition(1, 1);
    t.checkExpect(card.posn, new Posn(150, 55));
    card2.updatePosition(2, 2);
    t.checkExpect(card2.posn, new Posn(270, 90));
  }

  // tests the onMouseClicked method
  void testOnMouseClicked(Tester t) {
    GameWorld examplegameworld = new GameWorld();

    examplegameworld.game = new ConcentrationGame();
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(500, 500));
    examplegameworld.game.board = new ArrayList<ArrayList<Card>>(
        Arrays.asList(new ArrayList<Card>(Arrays.asList(card, card2))));
    examplegameworld.onMouseClicked(new Posn(0, 0));
    t.checkExpect(card.flip, true);
    t.checkExpect(card2.flip, false);

  }

  // tests the makeEndScene method
  void testMakeEndScene(Tester t) {
    GameWorld examplegameworld = new GameWorld();
    examplegameworld.timer = 2;
    examplegameworld.remaining = 2;
    WorldScene initialScene = new WorldScene(500, 500);
    initialScene.placeImageXY(new TextImage("You've Won", 60, Color.green), 250, 250);
    initialScene.placeImageXY(new TextImage("Time: 1", 40, Color.green), 250, 400);
    t.checkExpect(examplegameworld.makeEndScene(), initialScene);
    GameWorld examplegameworld2 = new GameWorld();
    examplegameworld2.timer = 0;
    examplegameworld2.remaining = 0;
    WorldScene initialScene2 = new WorldScene(500, 500);
    initialScene2.placeImageXY(new TextImage("You've Lost", 60, Color.red), 250, 250);
    initialScene2.placeImageXY(new TextImage("Time: 0", 40, Color.red), 250, 400);
    t.checkExpect(examplegameworld2.makeEndScene(), initialScene2);

  }

  // tests the worldEnds method
  void testWorldEnd(Tester t) {
    GameWorld examplegameworld = new GameWorld();
    examplegameworld.game = new ConcentrationGame();
    examplegameworld.game.createBoard(new Random(1));
    for (ArrayList<Card> ac : examplegameworld.game.board) {
      for (Card c : ac) {
        c.found = true;
      }
    }
    WorldScene initialScene = new WorldScene(500, 500);
    initialScene.placeImageXY(new TextImage("You've Won", 60, Color.green), 250, 250);
    initialScene.placeImageXY(new TextImage("Time: " + Integer.toString(0 / 2), 40, Color.green),
        250, 400);
    t.checkExpect(examplegameworld.worldEnds(), new WorldEnd(true, initialScene));
    GameWorld examplegameworld2 = new GameWorld();
    examplegameworld2.game = new ConcentrationGame();
    examplegameworld2.game.createBoard(new Random(1));
    t.checkExpect(examplegameworld2.worldEnds(),
        new WorldEnd(false, examplegameworld2.makeScene()));
    GameWorld examplegameworld3 = new GameWorld();
    examplegameworld3.game = new ConcentrationGame();
    examplegameworld3.game.createBoard(new Random(1));
    examplegameworld3.remaining = 0;
    WorldScene initialScene3 = new WorldScene(500, 500);
    initialScene3.placeImageXY(new TextImage("You've Lost", 60, Color.red), 250, 250);
    initialScene3.placeImageXY(new TextImage("Time: 0", 40, Color.red), 250, 400);
    t.checkExpect(examplegameworld3.worldEnds(), new WorldEnd(true, initialScene3));

  }

  // tests the onTick method
  void testOnTick(Tester t) {
    GameWorld examplegameworld = new GameWorld();
    examplegameworld.game = new ConcentrationGame();
    Card card = new Card(2, "♥", new Posn(0, 0));
    Card card2 = new Card(4, "♣", new Posn(500, 500));
    examplegameworld.game.board = new ArrayList<ArrayList<Card>>(
        Arrays.asList(new ArrayList<Card>(Arrays.asList(card, card2))));
    card2.flip = true;
    card.flip = true;
    examplegameworld.onTick();
    t.checkExpect(card.found, false);
    t.checkExpect(card2.found, false);
    t.checkExpect(examplegameworld.pairsleft, 26);
    card2.flip = true;
    card.flip = true;
    card2.rank = 2;
    card2.suit = "♥";
    card2.red = true;
    t.checkExpect(card.cardsEqual(card2), true);
    examplegameworld.onTick();
    t.checkExpect(card.found, true);
    t.checkExpect(card2.found, true);
    t.checkExpect(examplegameworld.pairsleft, 25);

  }
}
