import glfw
import pyrr
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
import random
import cmath

# A small tolerance for comparing floats for equality
TOL = 1.e-5
# psi = 1/phi where phi is the Golden ratio, sqrt(5)+1)/2
psi = (math.sqrt(5) - 1) / 2
# psi**2 = 1 - psi
psi2 = 1 - psi

class RobinsonTriangle:
    """
    A class representing a Robinson triangle and the rhombus formed from it.
    """

    def __init__(self, A, B, C):
        """
        Initialize the triangle with the ordered vertices. A and C are the
        vertices at the equal base angles; B is at the vertex angle.
        """
 
        self.A, self.B, self.C = A, B, C

    def centre(self):
        """
        Return the position of the centre of the rhombus formed from two
        triangles joined by their bases.
        """

        return (self.A + self.C) / 2

    def path(self, rhombus=True):
        """
        Return the SVG "d" path element specifier for the rhombus formed
        by this triangle and its mirror image joined along their bases. If
        rhombus=False, the path for the triangle itself is returned instead.
        """

        AB, BC = self.B - self.A, self.C - self.B 
        xy = lambda v: (v.real, v.imag)
        if rhombus:
            return 'm{},{} l{},{} l{},{} l{},{}z'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC) + xy(-AB))
        return 'm{},{} l{},{} l{},{}z'.format(*xy(self.A) + xy(AB)
                                                        + xy(BC))


    def get_arc_d(self, U, V, W, half_arc=False):
        """
        Return the SVG "d" path element specifier for the circular arc between
        sides UV and UW, joined at half-distance along these sides. If
        half_arc is True, the arc is at the vertex of a rhombus; if half_arc
        is False, the arc is drawn for the corresponding vertices of a
        Robinson triangle.
        """

        start = (U + V) / 2
        end = (U + W) / 2
        # arc radius
        r = abs((V - U) / 2)

        if half_arc:
            # Find the endpoint of the "half-arc" terminating on the triangle
            # base
            UN = V + W - 2*U
            end = U + r * UN / abs(UN)

        # ensure we draw the arc for the angular component < 180 deg
        cross = lambda u, v: u.real*v.imag - u.imag*v.real
        US, UE = start - U, end - U
        if cross(US, UE) > 0:
            start, end = end, start
        return 'M {} {} A {} {} 0 0 0 {} {}'.format(start.real, start.imag,
                                                    r, r, end.real, end.imag)

    def arcs(self, half_arc=False):
        """
        Return the SVG "d" path element specifiers for the two circular arcs
        about vertices A and C. If half_arc is True, the arc is at the vertex
        of a rhombus; if half_arc is False, the arc is drawn for the
        corresponding vertices of a Robinson triangle.
        """
        
        D = self.A - self.B + self.C
        arc1_d = self.get_arc_d(self.A, self.B, D, half_arc)
        arc2_d = self.get_arc_d(self.C, self.B, D, half_arc)
        return arc1_d, arc2_d

    def conjugate(self):
        """
        Return the vertices of the reflection of this triangle about the
        x-axis. Since the vertices are stored as complex numbers, we simply
        need the complex conjugate values of their values.
        """
        
        return self.__class__(self.A.conjugate(), self.B.conjugate(),
                              self.C.conjugate())

class BtileL(RobinsonTriangle):
    """
    A class representing a "B_L" Penrose tile in the P3 tiling scheme as
    a "large" Robinson triangle (sides in ratio 1:1:phi).
    """

    def inflate(self):
        """
        "Inflate" this tile, returning the three resulting Robinson triangles
        in a list.
        """

        # D and E divide sides AC and AB respectively
        D = psi2 * self.A + psi * self.C
        E = psi2 * self.A + psi * self.B
        # Take care to order the vertices here so as to get the right
        # orientation for the resulting triangles.
        return [BtileL(D, E, self.A),
                BtileS(E, D, self.B),
                BtileL(self.C, D, self.B)]

class BtileS(RobinsonTriangle):
    """
    A class representing a "B_S" Penrose tile in the P3 tiling scheme as
    a "small" Robinson triangle (sides in ratio 1:1:psi).
    """

    def inflate(self):
        """
        "Inflate" this tile, returning the two resulting Robinson triangles
        in a list.
        """
        D = psi * self.A + psi2 * self.B
        return [BtileS(D, self.C, self.A),
                BtileL(self.C, D, self.B)]

class PenroseP3:
    """ A class representing the P3 Penrose tiling. """

    def __init__(self, scale=200, ngen=4, config={}):
        """
        Initialise the PenroseP3 instance with a scale determining the size
        of the final image and the number of generations, ngen, to inflate
        the initial triangles. Further configuration is provided through the
        key, value pairs of the optional config dictionary.
        """
        
        self.scale = scale
        self.ngen = ngen

        # Default configuration
        self.config = {'width': '100%', 'height': '100%',
                       'stroke-colour': '#fff',
                       'base-stroke-width': 0.05,
                       'margin': 1.0,
                       'tile-opacity': 0.6,
                       'random-tile-colours': False,
                       'Stile-colour': '#08f',
                       'Ltile-colour': '#0035f3',
                       'Aarc-colour': '#f00',
                       'Carc-colour': '#00f',
                       'draw-tiles': True,
                       'draw-arcs': False,
                       'reflect-x': True,
                       'draw-rhombuses': True,
                       'rotate': 0,
                       'flip-y': False, 'flip-x': False,
                      }
        self.config.update(config)
        # And ensure width, height values are strings for the SVG
        self.config['width'] = str(self.config['width'])
        self.config['height'] = str(self.config['height'])

        self.elements = []

    def set_initial_tiles(self, tiles):
        self.elements = tiles

    def inflate(self):
        """ "Inflate" each triangle in the tiling ensemble."""
        new_elements = []
        for element in self.elements:
            new_elements.extend(element.inflate())
        self.elements = new_elements

    def remove_dupes(self):
        return
            

    def add_conjugate_elements(self):
        """ Extend the tiling by reflection about the x-axis. """

        self.elements.extend([e.conjugate() for e in self.elements])

    def rotate(self, theta):
        """ Rotate the figure anti-clockwise by theta radians."""

        rot = math.cos(theta) + 1j * math.sin(theta)
        for e in self.elements:
            e.A *= rot
            e.B *= rot
            e.C *= rot

    def flip_y(self):
        """ Flip the figure about the y-axis. """

        for e in self.elements:
            e.A = complex(-e.A.real, e.A.imag)
            e.B = complex(-e.B.real, e.B.imag)
            e.C = complex(-e.C.real, e.C.imag)

    def flip_x(self):
        """ Flip the figure about the x-axis. """

        for e in self.elements:
            e.A = e.A.conjugate()
            e.B = e.B.conjugate()
            e.C = e.C.conjugate()

    def make_tiling(self):
        """ Make the Penrose tiling by inflating ngen times. """

        for gen in range(self.ngen):
            self.inflate()
        if self.config['draw-rhombuses']:
            self.remove_dupes()
        if self.config['reflect-x']:
            self.add_conjugate_elements()
            self.remove_dupes()

        # Rotate the figure anti-clockwise by theta radians.
        theta = self.config['rotate']
        if theta:
            self.rotate(theta)

        # Flip the image about the y-axis (note this occurs _after_ any
        # rotation.
        if self.config['flip-y']:
            self.flip_y()

        # Flip the image about the x-axis (note this occurs _after_ any
        # rotation and after any flip about the y-axis.
        if self.config['flip-x']:
            self.flip_x()

    def get_tile_colour(self, e):
        """ Return a HTML-style colour string for the tile. """

        if self.config['random-tile-colours']:
            # Return a random colour as '#xxx'
            return '#' + hex(random.randint(0,0xfff))[2:]

        # Return the colour string, or call the colour function as appropriate
        if isinstance(e, BtileL):
            if hasattr(self.config['Ltile-colour'], '__call__'):
                return self.config['Ltile-colour'](e)
            return self.config['Ltile-colour']

        if hasattr(self.config['Stile-colour'], '__call__'):
                return self.config['Stile-colour'](e)
        return self.config['Stile-colour']



def main():



    scale = 100
    config={'tile-opacity': 0.9, 'stroke-colour': '#800',
            'Stile-colour': '#f00', 'Ltile-colour': '#ff0'}
    tiling = PenroseP3(scale*1.5, ngen=7, config=config)

    theta = math.pi / 5
    rot = math.cos(theta) + 1j*math.sin(theta)
    A1 = scale + 0.j
    B = 0 + 0j
    C1 = C2 = A1 * rot
    A2 = A3 = C1 * rot
    C3 = C4 = A3 * rot
    A4 = A5 = C4 * rot
    C5 = -A1
    tiling.set_initial_tiles([BtileS(A1, B, C1), BtileS(A2, B, C2),
                            BtileS(A3, B, C3), BtileS(A4, B, C4),
                            BtileS(A5, B, C5)])
    tiling.make_tiling()
    


    if not glfw.init():
        return

    width = 1200
    height = 1000

    window = glfw.create_window(width, height, "Penrose", None, None)

    if not window:
        glfw.terminate()
        return


    glfw.make_context_current(window)





    # convert to 32bit float

    verticesl = []
    verticess = []
    points = []
    colors = []
    for element in tiling.elements:
        points.append((element.A.real/100, element.A.imag/100))
        points.append((element.B.real/100, element.B.imag/100))
        points.append((element.C.real/100, element.C.imag/100))
        if isinstance(element, BtileL):
            verticesl.append(element.A.real/100)
            verticesl.append(element.A.imag/100)
            verticesl.append(element.B.real/100)
            verticesl.append(element.B.imag/100)
            verticesl.append(element.C.real/100)
            verticesl.append(element.C.imag/100)
        else:
            verticess.append(element.A.real/100)
            verticess.append(element.A.imag/100)
            verticess.append(element.B.real/100)
            verticess.append(element.B.imag/100)
            verticess.append(element.C.real/100)
            verticess.append(element.C.imag/100)

    
    verticesL = np.array(verticesl, dtype= np.float32)
    verticesS = np.array(verticess, dtype= np.float32)
    
    
    VERTEX_SHADER = """

        #version 330
        layout (location = 0) in vec3 position;
        uniform mat4 rotation;
        void main() {
        gl_Position = rotation * vec4(position.x * 1.5, position.y * 1.5, position.z * 1.5, 1.0);

    }


    """


    FRAGMENT_SHADER = """
        #version 330
        
        in vec4 col;
        in vec4 gl_FragCoord;
        out vec4 Color;
        uniform vec3 triangleColor;
        uniform float time;
        void main() {
        
        float value_X = abs(gl_FragCoord.x - 600);
        float value_Y = abs(gl_FragCoord.y - 500);
        
        
        if(time < 10){
            value_X = value_X * (150 - (time + 19) * 5.15);
            value_Y = value_Y * (150 - (time + 19) * 5.15);
        }
        else if(time < 15){
            value_X = 0;
            value_Y = 0;
        }
            
        
        if(value_X > 600){
            value_X = 600;
        }
        
        if(value_Y > 500){
            value_Y = 500;
        }
        
        
        Color = vec4((triangleColor.x)  * (1 - value_X/600) * (1 - value_Y/500), (triangleColor.y)  * (1 - value_X/600) * (1 - value_Y/500), (triangleColor.z)  * (1 - value_X/600) * (1 - value_Y/500),1.0f);

        }


    """

    # Compile The Program and shaders

    shader =  OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER,GL_VERTEX_SHADER),
                                             OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))



    #Create Buffers in gpu
    VBOS = glGenBuffers(1)
    VAOS = glGenVertexArrays(1)
    
    VBOL = glGenBuffers(1)
    VAOL = glGenVertexArrays(1)
    
    
    
   
    #Triangulos BtileS
    glBindVertexArray(VAOS)
    glBindBuffer(GL_ARRAY_BUFFER, VBOS)
    glBufferData(GL_ARRAY_BUFFER, verticesS.nbytes, verticesS, GL_STATIC_DRAW)
    
    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)
    
    #Triangulos BtileL
    glBindVertexArray(VAOL)
    glBindBuffer(GL_ARRAY_BUFFER, VBOL)
    glBufferData(GL_ARRAY_BUFFER, verticesL.nbytes, verticesL, GL_STATIC_DRAW)
    
    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)
    

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(0.0, 0.0 , 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        #get current run time in seconds for current loop
        currentTime = glfw.get_time()
        time = currentTime % 20
        #print(time)
        
        #create rotation matrixes
        rotZ1 = pyrr.Matrix44.from_z_rotation(currentTime/2)
        rotZ2 = pyrr.Matrix44.from_z_rotation(-1 * currentTime/2)
        rot_0 = pyrr.Matrix44.from_z_rotation(0)
        
        glUseProgram(shader)
        
        
        #draw filled triangles
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        #BtileS
        glBindVertexArray(VAOS)
        
        uniformTime = glGetUniformLocation(shader, "time");
        uniformRotation = glGetUniformLocation(shader, "rotation");
        uniform = glGetUniformLocation(shader, "triangleColor");
        
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ1)
        glUniform1f(uniformTime, time)
        
        glUniform3fv(uniform, 1, (1, 0, 0))
        glDrawArrays(GL_TRIANGLES, 0, len(verticesS))
        
        
        #BtileL
        glBindVertexArray(VAOL)
        
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ2)
        glUniform3fv(uniform, 1, (0, 0, 1))
        glDrawArrays(GL_TRIANGLES, 0, len(verticesL))

        glUniform3fv(uniform, 1, (0.0, 0.0, 0.0))
        
        
        #draw lines
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
        glLineWidth(2)
        
        #BtileS
        glBindVertexArray(VAOS)
        uniform = glGetUniformLocation(shader, "triangleColor");
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ1)
        glUniform3fv(uniform, 1, (0, 0, 0))
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glRotatef(30, 1, 0, 0)  
        
        glDrawArrays(GL_TRIANGLES, 0, len(verticesS))
        
        glPopMatrix()
        
              
        
        #BtileL
        glBindVertexArray(VAOL)
        glUniform3fv(uniform, 1, (0, 0, 0))
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ2)
        glDrawArrays(GL_TRIANGLES, 0, len(verticesL))

        glUniform3fv(uniform, 1, (0.0, 0.0, 0.0))
        
        #draw points
        glPolygonMode(GL_FRONT_AND_BACK,GL_POINT)
        glPointSize(3)
        
        #BtileS
        glBindVertexArray(VAOS)
        uniform = glGetUniformLocation(shader, "triangleColor");
        glUniform3fv(uniform, 1, (1, 1, 1))
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ1)
        glDrawArrays(GL_TRIANGLES, 0, len(verticesS))
        
        
        #BtileL
        glBindVertexArray(VAOL)
        glUniform3fv(uniform, 1, (1, 1, 1))
        glUniformMatrix4fv(uniformRotation, 1, GL_FALSE, rotZ2)
        glDrawArrays(GL_TRIANGLES, 0, len(verticesL))

        glUniform3fv(uniform, 1, (0.0, 0.0, 0.0))
        
        
        
        
        glfw.swap_buffers(window)


    glfw.terminate()







if __name__ == "__main__":
    main()