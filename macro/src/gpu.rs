use quote::ToTokens;
use syn::punctuated::Punctuated;

use crate::transformer::CommandsTransformer;

struct State {
    current_dispose_index: u32,
    dispose_forward_decl: proc_macro2::TokenStream,
    dispose_borrow_takes: proc_macro2::TokenStream,
    dispose_ret_expr: Punctuated<syn::Expr, syn::Token![,]>,

    recycled_state_count: usize,
}
impl Default for State {
    fn default() -> Self {
        Self {
            dispose_forward_decl: proc_macro2::TokenStream::new(),
            dispose_borrow_takes: proc_macro2::TokenStream::new(),
            dispose_ret_expr: Punctuated::new(),
            current_dispose_index: 0,
            recycled_state_count: 0,
        }
    }
}

impl CommandsTransformer for State {
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        todo!()
    }

    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let base = input.base.clone();

        let dispose_token_name =
            quote::format_ident!("__future_dispose_{}", self.current_dispose_index);
        self.current_dispose_index += 1;

        // Creates a location to store the dispose future. Dispose futures should be invoked
        // when the future returned by the parent QueueFutureBlock was invoked.
        // When the corresponding QueueFuture was awaited, the macro writes the return value of
        // its dispose method into this location.
        // This needs to be a cell because __dispose_fn_future "pre-mutably-borrows" the value.
        // This value also needs to be written by the .await statement, creating a double borrow.
        self.dispose_forward_decl.extend(quote::quote! {
            let mut #dispose_token_name = ::std::cell::Cell::new(None);
        });
        // The program may return at different locations, and upon return, not all futures may have
        // been awaited. We should only await the dispose futures for those actually awaited so far
        // in the QueueFuture. We check this at runtime to avoid returning different variants of
        // the dispose future.
        self.dispose_ret_expr
            .push(syn::Expr::Verbatim(dispose_token_name.to_token_stream()));
        // __dispose_fn_future borrows #dispose_token_name ahead of time. We move out the values
        // from the cell when __dispose_fn_future was invoked so that we can return a Future with
        // 'static lifetime.
        self.dispose_borrow_takes.extend(quote::quote! {
            let #dispose_token_name = #dispose_token_name.replace(None);
        });

        let index = syn::Index::from(self.recycled_state_count);
        self.recycled_state_count += 1;
        syn::Expr::Verbatim(quote::quote! {{
            let mut fut = #base;
            let mut fut_pinned = unsafe{std::pin::Pin::new_unchecked(&mut fut)};
            fut_pinned.as_mut().init(
                unsafe{&mut *(__ctx as *mut ::async_ash::queue::SubmissionContext)},
                &mut unsafe{&mut *__recycled_states}.#index,
                __current_queue
            );
            let output = loop {
                match fut_pinned.as_mut().record(
                    unsafe{&mut *(__ctx as *mut ::async_ash::queue::SubmissionContext)},
                    &mut unsafe{&mut *__recycled_states}.#index
                ) {
                    ::async_ash::queue::QueueFuturePoll::Ready { next_queue, output } => {
                        __current_queue = next_queue;
                        break output;
                    },
                    ::async_ash::queue::QueueFuturePoll::Semaphore => (__initial_queue, __ctx, __recycled_states) = yield true,
                    ::async_ash::queue::QueueFuturePoll::Barrier => (__initial_queue, __ctx, __recycled_states) = yield false,
                };
            };
            #dispose_token_name.replace(Some(fut.dispose()));
            output
        }})
    }
    //               queue
    // Leaf  nodes   the actual queue
    // Join  nodes   None, or the actual queue if all the same.
    // Block nodes   None
    // for blocks, who yields initially?
    // inner block yields. outer block needs to give inner block the current queue, and the inner block choose to yield or not.

    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        syn::Expr::Macro(mac.clone())
    }

    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        let returned_item = ret
            .expr
            .as_ref()
            .map(|a| *a.clone())
            .unwrap_or(syn::Expr::Verbatim(quote::quote!(())));
        let dispose_ret_expr = &self.dispose_ret_expr;

        let token_stream = quote::quote!(
            {
                return (__current_queue, (#dispose_ret_expr), #returned_item);
            }
        );
        Some(syn::Expr::Verbatim(token_stream))
    }
}

pub fn proc_macro_gpu(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = State::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();

    let dispose_forward_decl = state.dispose_forward_decl;
    let dispose_ret_expr = state.dispose_ret_expr;
    let dispose_borrow_takes = state.dispose_borrow_takes;

    if let Some(last) = inner_closure_stmts.last_mut() {
        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!({
                return (__current_queue, (#dispose_ret_expr), #expr);
            });
            *expr = syn::Expr::Verbatim(token_stream);
        } else {
            let token_stream = quote::quote!({
                return (__current_queue,  (#dispose_ret_expr), ());
            });
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Verbatim(token_stream)))
        }
    }
    let recycled_states_type = syn::Type::Tuple(syn::TypeTuple {
        paren_token: Default::default(),
        elems: {
            let mut elems = syn::punctuated::Punctuated::from_iter(
                std::iter::repeat(syn::Type::Infer(syn::TypeInfer {
                    underscore_token: Default::default(),
                }))
                .take(state.recycled_state_count),
            );
            if state.recycled_state_count == 1 {
                elems.push_punct(Default::default());
            }
            elems
        },
    });
    quote::quote! {
        async_ash::queue::QueueFutureBlock::new(static |(mut __initial_queue, mut __ctx, mut __recycled_states):(_,_,*mut #recycled_states_type)| {
            #dispose_forward_decl
            let mut __current_queue: ::async_ash::queue::QueueMask = __initial_queue;
            #(#inner_closure_stmts)*
        })
    }
}
