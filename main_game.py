import pygame
import pygame_widgets
import pygame_widgets.button as btn, pygame_widgets.textbox as txbx
import os
#-------------------------------------------------------------------------------------
import itertools
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Iterable, Optional, Union

_all_numpy_int_types = Union[np.int8, np.int16, np.int32, np.int64]

X = 1
O = 2

#--------------------------------------------------------------------------------------------------------
class Board:
    #preparation :
            #? the coordination of the board : top left is (0, 0) and the coordinates be like (column, line) increasing left to right and top to buttom
    def __init__(self, dimensions: Iterable[int] = (3, 3), x_in_a_row: int = 3) -> None:
        """
        TicTacToe board.
        :param dimensions: The dimensions of the board.
        :param x_in_a_row: How many marks in a row are needed to win.
        """
        self.dimensions = tuple(dimensions)
        self.x_in_a_row = x_in_a_row
        self.board = self.create_board()
        self._directions = self.find_directions()
        self.move_count = 0
        self.moves: List[Move] = []
        self.x: List[Move] = []
        self.o: List[Move] = []
        self.turn = X
    #board creation (called automatically)
    def create_board(self) -> npt.NDArray[np.int8]:
        """
        Create the board state.
        :return: A `numpy.ndarray` filled with 0s.
        """
        return np.zeros(self.dimensions, dtype=np.int8)
    #make an other copy of the board :
    def copy(self) :
        """
        Get a copy of the board.
        :return: A copy of the board.
        """
        board = Board(self.dimensions, self.x_in_a_row)
        board.turn = self.turn
        board.board = self.board.copy()
        return board
    #function return the mark at a given position :
    def get_mark_at_position(self, position: Iterable[int]) -> int:
        """
        Get the mark at a position.
        :param position: The position to check.
        :return: The player that has a mark at the position.
        """
        position = tuple(position)
        return int(self.board[position])
    #set a move (mark) in the board and update it with no condition :
    def set_mark(self, coordinates: Iterable[int], player: int) -> None:
        """
        Set a mark at a position.
        :param coordinates: The position to add a mark at.
        :param player: The player that put the mark at the position.
        """
        self.board[tuple(coordinates)] = player
        if player == X:
            self.x.append(Move(coordinates))
        else:
            self.o.append(Move(coordinates))
    #returns boolean indicates if the position is empty :
    def is_empty(self, position: Iterable[int]) -> bool:
        """
        Get if a position is empty.
        :param position: The position to check.
        :return: If the position is empty.
        """
        return self.get_mark_at_position(position) == 0
    #make a move in the board (more efficient) :
    def push(self, coordinates: Iterable[int]) -> None:
        """
        Push a move.
        :param coordinates: The position to add a mark at.
        """
        coordinates = tuple(coordinates)
        if not self.is_empty(coordinates):
            raise ValueError("Position is not empty.")
        move = Move(coordinates)
        self.set_mark(coordinates, self.turn)
        self.turn = X if self.turn == O else O
        self.moves.append(move)
        self.move_count += 1

    def find_directions(self) -> List[Tuple[int, ...]]:
        """
        Get directions to be used when checking for a win.
        :return: The directions to check for a win.
        """
        directions = list(itertools.product([1, 0, -1], repeat=len(self.dimensions)))
        correct_directions = []
        for direction in directions:
            for item in direction:
                if item > 0:
                    correct_directions.append(direction)
                    break
                elif item < 0:
                    break
        return correct_directions
    #return all the possible moves in the board :
    def possible_moves(self) -> npt.NDArray[np.int64]:
        """
        Get all possible moves.
        :return: All the positions where there is no mark.
        """
        return np.argwhere(self.board == 0)

    def out_of_bounds(self, pos: npt.NDArray[_all_numpy_int_types]) -> np.bool_:
        """
        Get if a position is out of the board.
        :param pos: The position to check.
        :return: If the position is out of the board.
        """
        return (pos < 0).any() or (pos >= self.dimensions).any()

    def in_bounds(self, pos: npt.NDArray[_all_numpy_int_types]) -> bool:
        """
        Get if a position is inside the board.
        :param pos: The position to check.
        :return: If the position is inside the board.
        """
        return not self.out_of_bounds(pos)

    def has_won(self, player: int) -> bool:
        """
        Get if a player has won.
        :param player: The player to check.
        :return: If the player has won.
        """
        positions = np.argwhere(self.board == player)
        for position in positions:
            for direction in self._directions:
                for in_a_row in range(1, self.x_in_a_row):
                    pos = position + np.multiply(direction, in_a_row)
                    if self.out_of_bounds(pos) or self.board[tuple(pos)] != player:
                        break
                else:
                    return True
        return False

    def result(self) -> Optional[int]:
        """
        Get the result of the game.
        :return: The result of the board.
        """
        x_won = self.has_won(X)
        o_won = self.has_won(O)
        if x_won and o_won:
            raise Exception(f"Both X and O have {self.x_in_a_row} pieces in a row.")
        elif x_won:
            return X
        elif o_won:
            return O
        elif self.board.all():
            return 0
        return None

    def _get_dimension_repr(self, board_partition: npt.NDArray[np.int8]) -> str:
        """
        Get a visual representation of a part of the board.
        :param board_partition: A part of the board.
        :return: A visual representation of a part of the board.
        """
        if len(board_partition.shape) > 1:
            board_repr = ""
            divider = ((board_partition.shape[0] * 4 - 1) * "-" + "\n") * (len(board_partition.shape) - 1)
            for board_partition_index in range(board_partition.shape[-1]):
                board_repr += self._get_dimension_repr(board_partition[..., board_partition_index]) + "\n"
                board_repr += divider
            board_repr = board_repr[:-(len(divider) + 1)]
            return board_repr
        else:
            row = ""
            for item in board_partition:
                mark = "O" if item == O else ("X" if item == X else " ")
                row += f" {mark} |"
            row = row[:-1]
            return row

    def __repr__(self) -> str:
        """
        Get a visual representation of the board.
        :return: A visual representation of the board.
        """
        return self._get_dimension_repr(self.board)


class Move:
    def __init__(self, coordinate_move: Optional[Iterable[int]] = None, str_move: Optional[str] = None) -> None:
        """
        Convert a move to other types.
        :param coordinate_move: A `tuple` with the position of the move.
        :param str_move: A move that is in a human-readable format.
        """
        assert coordinate_move or str_move
        self.coordinate_move = coordinate_move
        self.str_move = str_move
        if self.coordinate_move:
            self.str_move = "-".join(map(str, tuple(self.coordinate_move)))
        elif self.str_move:
            self.coordinate_move = tuple(map(int, self.str_move.split("-")))
#---------------------------------------------------------------------------------------------------------------------------------
pygame.init()
pygame.mixer.init()
path = os.path.dirname(__file__)
click_sfx = pygame.mixer.Sound(os.path.join(path,"sound\click.wav"))
clock = pygame.time.Clock()
FPS = 30
pygame.font.init()
font = pygame.font.SysFont(name="Times New Roman", size = 40)
text_lines = font.render("N° of lines",True,(255,255,255))
text_cols = font.render("N° of cols",True,(255,255,255))
text_chain = font.render("chain winning",True,(255,255,255))
x_winner = font.render("player with X has won",True,(0,0,0))
o_winner = font.render("player with O has won",True,(0,0,0))
draw = font.render("the game ended in a draw",True,(0,0,0))
author1 = font.render("game logic and graphical UI",True,(0,0,0))
author2 = font.render("developped by B.Beligh",True,(0,0,0))

length, height = 712, 712
window = pygame.display.set_mode((length, height))
board = None

lines, cols, chain = 3, 3, 3
##################################################################################
#? this function will assigned the params given in the widgets to the actual board
def get_size() :
    loop = False
    global lines, cols, chain, board, board_squares
    try :
        lines = int(lines_widget.getText())
        cols = int(columns_widget.getText())
        chain = int(chain_widget.getText())
        board = Board((lines,cols),chain)
        board_squares = make_board(cols,lines)
    except Exception :
        print("error, one or all the parametres is(are) wrong!!")
        loop = True
    global update
    if not loop :
        update=False
    window.fill(square((0,0),(1,1),window).color)
def new_game():
    global update, ok
    update = True
    ok = False
    get_size()
##################################################################################
#? button configuration parametres :
confirmation_button_size = (150, 50)
low_difference_of_th_button = 250
confirmation_button = btn.Button(window, length/2-confirmation_button_size[0]/2, height-low_difference_of_th_button-confirmation_button_size[1], 
                                 confirmation_button_size[0], confirmation_button_size[1], text="confirmation", onClick=get_size,
                                 font=pygame.font.SysFont(name="Times New Roman", size = 25), radius=10)
ok_button = btn.Button(window, -100, -100, confirmation_button_size[0], confirmation_button_size[1], text="new game", onClick=new_game)
#? labels configuration parametres :
left_edge_diff = 100
vertical_pos = 200
widget_size = (100, 50)
columns_widget = txbx.TextBox(window, left_edge_diff, vertical_pos, widget_size[0], widget_size[1], fontSize=30)
lines_widget = txbx.TextBox(window, length/2-widget_size[0]/2, vertical_pos, widget_size[0], widget_size[1], fontSize=30)
chain_widget = txbx.TextBox(window, length-left_edge_diff-widget_size[0], vertical_pos, widget_size[0], widget_size[1], fontSize=30)
columns_widget.fontSize = 3
lines_widget.setText("3")
columns_widget.setText("3")
chain_widget.setText("3")

class square :
    def __init__(self, position: tuple, board_size:tuple, window) :
        """ position in the position of the square in the board : like it's line and row references
            board_size is the number of lines and rows in the board whiche the square is placed
        """
        self.position = position
        self.size = [0, 0]
        self.size[0] = 712/board_size[0]
        self.size[1] = 712/board_size[1]
        self.width = self.size[0]
        self.height = self.size[1]
        self.x = self.position[0]*self.width
        self.y = self.position[1]*self.height
        self.center = (self.x+self.width/2, self.y+self.height/2)
        self.window = window
        self.hover = self.X = self.O = False
        if ((position[0]+position[1]) % 2) == 0 :
            self.set_color((121, 102, 244))
        else :
            self.set_color((208, 208, 208))

    def set_color (self, color:tuple):
        self.rectangle = pygame.Rect(self.x, self.y, self.width, self.height)
        self.color = color

    def manage_hovering(self, event):
        if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP :
            x, y= event.pos
            if (x >= self.x and x <= self.x+self.width) and (y >= self.y and y <= self.y+self.height):
                self.hover = True
            else :
                self.hover = False

    def manage_set_mark(self, board):
        self.mark = board.get_mark_at_position(self.position)
        if self.mark == 2 : #which mensa it's an o
            self.O = True
        if self.mark == 1 : #which means it's an X
            self.X = True
    def update(self, event, board):
        self.manage_set_mark(board)
        self.manage_hovering(event)
        self.draw()
    def draw(self) :
        pygame.draw.rect(self.window, self.color, self.rectangle)
        if self.X:
            cst = 1/15
            hmarg = self.width/10
            vmarg = self.height/10
            length = self.width/8
            #determining the coordinates of all the points of the X and draw it as a polygone
            p1 = (self.x+hmarg, self.y+vmarg)
            p2 = (self.x+hmarg+length, self.y+vmarg)
            p3 = (self.x+self.width/2, self.y+self.height/2-self.height*cst)
            p4 = (self.x+self.width-hmarg-length, self.y+vmarg)
            p5 = (self.x+self.width-hmarg, self.y+vmarg)
            p6 = (self.x+self.width/2+self.width*cst, self.y+self.height/2)
            p7 = (self.x+self.width-hmarg, self.y+self.height-vmarg)
            p8 = (self.x+self.width-hmarg-length, self.y+self.height-vmarg)
            p9 = (self.x+self.width/2, self.y+self.height/2+self.height*cst)
            p10 = (self.x+hmarg+length, self.y+self.height-vmarg)
            p11 = (self.x+hmarg, self.y+self.height-vmarg)
            p12 = (self.x+self.width/2-self.width*cst, self.y+self.height/2)
            pygame.draw.polygon(window, (120,120,120),[p1 ,p2 ,p3 ,p4 ,p5 ,p6 ,p7 ,p8 ,p9 ,p10,p11,p12])
        if self.O :
            outter_rect = pygame.Rect(self.x+self.width/10, self.y+self.height/10, self.width*8/10, self.height*8/10)
            inner_rect = pygame.Rect(self.x+self.width/5, self.y+self.height/5, self.width*3/5, self.height*3/5)
            pygame.draw.ellipse(window, (120,120,120),outter_rect)
            pygame.draw.ellipse(window, self.color, inner_rect)
#function to build the graphic of the board :
def make_board(lines, cols):
    board_squares = []
    board_element = []
    for line in range(lines):
        for col in range(cols):
            board_element.append(square((col,line), (cols,lines), window))
        board_squares.append(board_element)
        board_element = []
    return(board_squares)
#function to assign a mark to the hovered square of the board :
def make_mark (event: pygame.event):
    if event.type == pygame.MOUSEBUTTONDOWN :
        global board_squares, board
        for sq in board_squares :
            for sqare in sq:
                if sqare.hover and board.get_mark_at_position(sqare.position) not in (1,2) and not board.has_won(X) and not board.has_won(O):
                    board.push(sqare.position)
                    click_sfx.play()
board_squares = None
update = True
chosing_run = True
found = False
ok = False
#loop of the params choices
while chosing_run :
    events = pygame.event.get()
    for event in events :
        if event.type == pygame.QUIT :
            chosing_run = False
        if not update:
            make_mark(event)	
        if board_squares :
            for ss in board_squares :
                for s in ss:
                    s.update(event, board)
                    s.draw()
        if not update and board.has_won(X) :
            window.blit(x_winner, ((712-x_winner.get_width())/2, 356-x_winner.get_height()))
            ok = True
        elif not update and board.has_won(O):
            window.blit(o_winner, ((712-o_winner.get_width())/2, 356-o_winner.get_height()))
            ok = True
        elif not update and board.result() == 0 :
            window.blit(draw, ((712-o_winner.get_width())/2, 356-o_winner.get_height()))
            ok = True
    if update :
        window.fill((0,36,81))
        window.blit(text_lines, (lines_widget.getX()-text_lines.get_width()/4, lines_widget.getY()+lines_widget.getHeight()))
        window.blit(text_cols, (columns_widget.getX()-text_cols.get_width()/4, columns_widget.getY()+columns_widget.getHeight()))
        window.blit(text_chain, (chain_widget.getX()-text_chain.get_width()/4, chain_widget.getY()+chain_widget.getHeight()))
        window.blit(author1,(356-author1.get_width()/2, 556))
        window.blit(author2,(356-author2.get_width()/2, 556+author2.get_height()))
        pygame_widgets.update(events)
    if ok :
        ok_button.setX(length/2-confirmation_button_size[0]/2)
        ok_button.setY(height-low_difference_of_th_button-confirmation_button_size[1])
        pygame_widgets.update(events)
    else :
        ok_button.setY(-100)
        ok_button.setX(-100)
    clock.tick(FPS)
    pygame.display.update()
pygame.quit()
