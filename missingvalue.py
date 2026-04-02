class Node:
    def __init__ (self,data=None,next=None):
        self.data=data
        self.next=next
class linkedlist:
    def __init__ (self,head):
        self.head=head
    def insert(self,data):
        node = Node(data, self.head)